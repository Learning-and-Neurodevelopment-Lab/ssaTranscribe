#!/usr/bin/env python3
"""
Batch Orchestrator for Large-Scale Audio Processing
Robust against interruptions, disconnections, and server issues
"""

import os
import sys
import json
import logging
import subprocess
import argparse
import signal
import atexit
import fcntl
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import time
import socket
import traceback

STAGE_ENVS = {
    1: "whisperx_cpu",   # Stage 1: WhisperX
    2: "pyannote_cpu",   # Stage 2: pyannote/NeMo
    3: "pyannote_cpu"   # Stage 3: fact-check stage (can share env or separate)
}

# Set up logging with file output
def setup_logging(log_dir: Path):
    """Setup dual logging to console and file"""
    log_file = log_dir / f"orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_file

logger = logging.getLogger(__name__)


class BatchOrchestrator:
    """Orchestrates batch processing with robust error handling"""
    
    def __init__(self, 
                 input_dir: str,
                 output_base_dir: str,
                 stage_scripts: Dict[str, str],
                 file_extensions: List[str] = ['.wav', '.mp3', '.m4a', '.mp4', '.mkv', '.avi', '.mov', '.webm'],
                 checkpoint_interval: int = 1,
                 retry_failed: int = 3,
                 enable_heartbeat: bool = True):
        
        self.input_dir = Path(input_dir)
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        self.stage_scripts = stage_scripts
        self.file_extensions = file_extensions
        self.checkpoint_interval = checkpoint_interval
        self.retry_failed = retry_failed
        self.enable_heartbeat = enable_heartbeat
        
        # Progress tracking files
        self.progress_file = self.output_base_dir / "batch_progress.json"
        self.lock_file = self.output_base_dir / "orchestrator.lock"
        self.heartbeat_file = self.output_base_dir / "heartbeat.txt"
        self.summary_file = self.output_base_dir / "batch_summary.txt"
        self.checkpoint_file = self.output_base_dir / "checkpoint.json"
        
        # Runtime state
        self.lock_fd = None
        self.shutdown_requested = False
        self.current_episode = None
        self.start_time = time.time()
        
        # Setup graceful shutdown handlers
        self._setup_signal_handlers()
        atexit.register(self._cleanup)
        
        # Acquire lock to prevent multiple instances
        self._acquire_lock()
        
        # Setup logging
        self.log_file = setup_logging(self.output_base_dir)
        logger.info(f"Logging to: {self.log_file}")
        
        # Load or initialize progress
        self.progress = self._load_progress()
        
        # Start heartbeat if enabled
        if self.enable_heartbeat:
            self._update_heartbeat("Initialized")
    
    def _setup_signal_handlers(self):
        """Setup handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            logger.warning(f"\n{'='*60}")
            logger.warning(f"Received signal: {signal_name}")
            logger.warning(f"Initiating graceful shutdown...")
            logger.warning(f"{'='*60}")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # kill
        try:
            signal.signal(signal.SIGHUP, signal_handler)   # Terminal closed
        except AttributeError:
            pass  # Windows doesn't have SIGHUP
    
    def _acquire_lock(self):
        """Acquire lock file to prevent multiple instances"""
        try:
            self.lock_fd = open(self.lock_file, 'w')
            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.lock_fd.write(f"{os.getpid()}\n")
            self.lock_fd.write(f"{datetime.now().isoformat()}\n")
            self.lock_fd.write(f"{socket.gethostname()}\n")
            self.lock_fd.flush()
            logger.info("Lock acquired - no other instances running")
        except IOError:
            logger.error("=" * 60)
            logger.error("ERROR: Another orchestrator instance is already running!")
            logger.error(f"Lock file: {self.lock_file}")
            logger.error("If you're sure no other instance is running, delete the lock file.")
            logger.error("=" * 60)
            sys.exit(1)
    
    def _release_lock(self):
        """Release lock file"""
        if self.lock_fd:
            try:
                fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
                self.lock_fd.close()
                self.lock_file.unlink(missing_ok=True)
                logger.info("Lock released")
            except Exception as e:
                logger.warning(f"Error releasing lock: {e}")
    
    def _cleanup(self):
        """Cleanup on exit"""
        if hasattr(self, 'shutdown_requested') and self.shutdown_requested:
            logger.info("Cleanup: Saving final state...")
            self._save_progress()
            self._save_checkpoint()
        self._release_lock()
    
    def _update_heartbeat(self, status: str):
        """Update heartbeat file to show process is alive"""
        if not self.enable_heartbeat:
            return
        
        try:
            with open(self.heartbeat_file, 'w') as f:
                f.write(f"PID: {os.getpid()}\n")
                f.write(f"Hostname: {socket.gethostname()}\n")
                f.write(f"Status: {status}\n")
                f.write(f"Current Episode: {self.current_episode or 'None'}\n")
                f.write(f"Last Update: {datetime.now().isoformat()}\n")
                f.write(f"Runtime: {(time.time() - self.start_time)/3600:.1f} hours\n")
                f.write(f"Progress: {self.progress.get('completed', 0)}/{self.progress.get('total', 0)}\n")
        except Exception as e:
            logger.debug(f"Failed to update heartbeat: {e}")
    
    def _load_progress(self) -> Dict:
        """Load progress from previous run or create new"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    logger.info(f"Resuming previous run:")
                    logger.info(f"  Completed: {progress['completed']}/{progress['total']}")
                    logger.info(f"  Failed: {progress['failed']}")
                    logger.info(f"  Started: {progress['started']}")
                    return progress
            except Exception as e:
                logger.warning(f"Could not load progress file: {e}")
                logger.info("Starting fresh...")
        
        return {
            'started': datetime.now().isoformat(),
            'total': 0,
            'completed': 0,
            'failed': 0,
            'episodes': {},
            'version': '2.0'
        }
    
    def _save_progress(self):
        """Save current progress atomically"""
        try:
            self.progress['last_updated'] = datetime.now().isoformat()
            
            # Atomic write: write to temp file, then rename
            temp_file = self.progress_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
            
            temp_file.replace(self.progress_file)
            logger.debug("Progress saved")
            
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def _save_checkpoint(self):
        """Save detailed checkpoint for recovery"""
        try:
            checkpoint = {
                'timestamp': datetime.now().isoformat(),
                'current_episode': self.current_episode,
                'runtime_hours': (time.time() - self.start_time) / 3600,
                'progress': self.progress
            }
            
            temp_file = self.checkpoint_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            temp_file.replace(self.checkpoint_file)
            logger.debug("Checkpoint saved")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _find_audio_files(self) -> List[Path]:
        """Find all audio/video files in input directory"""
        media_files = []
        for ext in self.file_extensions:
            media_files.extend(self.input_dir.glob(f"**/*{ext}"))
        
        media_files.sort()
        logger.info(f"Found {len(media_files)} media files (audio + video)")
        return media_files
    
    def _run_stage(self, stage_num: int, script_path: str, args: List[str], 
                   episode_id: str, attempt: int = 1) -> bool:
        """Run a single stage in its designated conda environment with retry logic"""
        stage_name = f"Stage {stage_num}"
        env_name = STAGE_ENVS.get(stage_num, None)
    
        if attempt > 1:
            logger.info(f"  {stage_name}: Retry attempt {attempt}/{self.retry_failed}")
        else:
            logger.info(f"  {stage_name}: Starting...")

        # Build command
        if env_name:
            cmd = ["conda", "run", "--no-capture-output", "-n", env_name, "python", script_path] + args
        else:
            cmd = ["python3", script_path] + args
        
        try:
            # Update heartbeat before starting stage
            self._update_heartbeat(f"Running {stage_name} on {episode_id}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout per stage
            )
            
            if result.returncode == 0:
                logger.info(f"  {stage_name}: ✓ Success")
                return True
            else:
                logger.error(f"  {stage_name}: ✗ Failed (exit code {result.returncode})")
                logger.error(f"    stderr: {result.stderr[:500]}")
                
                # Retry logic
                if attempt < self.retry_failed:
                    logger.info(f"  {stage_name}: Waiting 30s before retry...")
                    time.sleep(30)
                    return self._run_stage(stage_num, script_path, args, episode_id, attempt + 1)
                
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"  {stage_name}: ✗ Timeout (>2 hours)")
            
            if attempt < self.retry_failed:
                logger.info(f"  {stage_name}: Retrying after timeout...")
                return self._run_stage(stage_num, script_path, args, episode_id, attempt + 1)
            
            return False
            
        except Exception as e:
            logger.error(f"  {stage_name}: ✗ Exception: {e}")
            logger.debug(traceback.format_exc())
            
            if attempt < self.retry_failed:
                logger.info(f"  {stage_name}: Retrying after exception...")
                time.sleep(10)
                return self._run_stage(stage_num, script_path, args, episode_id, attempt + 1)
            
            return False
    
    def process_episode(self, media_file: Path, force: bool = False) -> bool:
        """Process a single episode through all 3 stages"""
        
        episode_id = media_file.stem
        self.current_episode = episode_id
        output_dir = self.output_base_dir / episode_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for shutdown request
        if self.shutdown_requested:
            logger.warning(f"[{episode_id}] Shutdown requested - stopping")
            return False
        
        # Check if already completed
        if not force and episode_id in self.progress['episodes']:
            status = self.progress['episodes'][episode_id]
            if status.get('completed', False):
                logger.info(f"[{episode_id}] Already completed - skipping")
                return True
            
            # Resume from failed stage
            start_stage = status.get('last_completed_stage', 0) + 1
            logger.info(f"[{episode_id}] Resuming from Stage {start_stage}")
        else:
            start_stage = 1
            self.progress['episodes'][episode_id] = {
                'media_file': str(media_file),
                'output_dir': str(output_dir),
                'started': datetime.now().isoformat(),
                'last_completed_stage': 0,
                'completed': False,
                'attempts': 0
            }
        
        logger.info(f"[{episode_id}] Processing: {media_file.name}")
        self.progress['episodes'][episode_id]['attempts'] += 1
        
        # Stage 1: WhisperX (handles both audio and video)
        if start_stage <= 1:
            if self.shutdown_requested:
                return False
            
            success = self._run_stage(
                1,
                self.stage_scripts['stage1'],
                [str(media_file), "--output-dir", str(output_dir)],
                episode_id
            )
            
            if not success:
                self.progress['episodes'][episode_id]['failed_stage'] = 1
                self.progress['episodes'][episode_id]['failed_at'] = datetime.now().isoformat()
                self._save_progress()
                return False
            
            self.progress['episodes'][episode_id]['last_completed_stage'] = 1
            self._save_progress()
        
        # Stage 2: NeMo (uses extracted audio from Stage 1)
        if start_stage <= 2:
            if self.shutdown_requested:
                return False
            
            # For Stage 2, we need to pass the audio file (extracted by Stage 1)
            # Check if audio.wav exists in output dir, otherwise use original media file
            extracted_audio = output_dir / "audio.wav"
            audio_for_stage2 = extracted_audio if extracted_audio.exists() else media_file
            
            success = self._run_stage(
                2,
                self.stage_scripts['stage2'],
                [str(audio_for_stage2), "--output-dir", str(output_dir)],
                episode_id
            )
            
            if not success:
                self.progress['episodes'][episode_id]['failed_stage'] = 2
                self.progress['episodes'][episode_id]['failed_at'] = datetime.now().isoformat()
                self._save_progress()
                return False
            
            self.progress['episodes'][episode_id]['last_completed_stage'] = 2
            self._save_progress()
        
        # Stage 3: Fact Check
        if start_stage <= 3:
            if self.shutdown_requested:
                return False
            
            success = self._run_stage(
                3,
                self.stage_scripts['stage3'],
                ["--output-dir", str(output_dir)],
                episode_id
            )
            
            if not success:
                self.progress['episodes'][episode_id]['failed_stage'] = 3
                self.progress['episodes'][episode_id]['failed_at'] = datetime.now().isoformat()
                self._save_progress()
                return False
            
            self.progress['episodes'][episode_id]['last_completed_stage'] = 3
            self._save_progress()
        
        # Mark as completed
        self.progress['episodes'][episode_id]['completed'] = True
        self.progress['episodes'][episode_id]['completed_at'] = datetime.now().isoformat()
        self.progress['completed'] += 1
        self._save_progress()
        self._save_checkpoint()
        
        logger.info(f"[{episode_id}] ✓ All stages complete")
        return True
    
    def run(self, force: bool = False, max_episodes: Optional[int] = None):
        """Run batch processing on all episodes"""
        
        media_files = self._find_audio_files()
        
        if not media_files:
            logger.error("No media files found!")
            return
        
        if max_episodes:
            media_files = media_files[:max_episodes]
            logger.info(f"Limiting to first {max_episodes} episodes")
        
        self.progress['total'] = len(media_files)
        self._save_progress()
        
        logger.info("=" * 60)
        logger.info(f"BATCH PROCESSING: {len(media_files)} episodes")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_base_dir}")
        logger.info(f"Retry attempts: {self.retry_failed}")
        logger.info(f"Checkpoint interval: {self.checkpoint_interval} episode(s)")
        logger.info("=" * 60)
        
        self.start_time = time.time()
        episodes_processed = 0
        
        for i, media_file in enumerate(media_files, 1):
            if self.shutdown_requested:
                logger.warning("\n" + "="*60)
                logger.warning("GRACEFUL SHUTDOWN INITIATED")
                logger.warning("="*60)
                logger.warning(f"Processed {episodes_processed} episodes before shutdown")
                logger.warning(f"Progress saved. Re-run to resume from here.")
                logger.warning("="*60)
                break
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Episode {i}/{len(media_files)}")
            logger.info(f"{'='*60}")
            
            self._update_heartbeat(f"Processing episode {i}/{len(media_files)}")
            
            success = self.process_episode(media_file, force=force)
            episodes_processed += 1
            
            if not success:
                self.progress['failed'] += 1
                self._save_progress()
                logger.warning(f"Episode failed, continuing to next...")
            
            # Periodic checkpoint
            if episodes_processed % self.checkpoint_interval == 0:
                self._save_checkpoint()
                logger.info(f"Checkpoint saved (every {self.checkpoint_interval} episodes)")
            
            # Progress update
            elapsed = time.time() - self.start_time
            rate = episodes_processed / elapsed if elapsed > 0 else 0
            remaining = (len(media_files) - i) / rate if rate > 0 else 0
            
            logger.info(f"\nProgress: {i}/{len(media_files)} episodes processed")
            logger.info(f"Success: {self.progress['completed']}, Failed: {self.progress['failed']}")
            logger.info(f"Rate: {rate*3600:.1f} episodes/hour")
            logger.info(f"Est. remaining time: {remaining/3600:.1f} hours")
        
        # Final summary
        self._write_summary()
        self._update_heartbeat("Completed")
        
        logger.info("\n" + "=" * 60)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total: {len(media_files)}")
        logger.info(f"Completed: {self.progress['completed']}")
        logger.info(f"Failed: {self.progress['failed']}")
        logger.info(f"Total time: {(time.time() - self.start_time)/3600:.1f} hours")
        logger.info(f"Summary: {self.summary_file}")
        logger.info(f"Full log: {self.log_file}")
        logger.info("=" * 60)
    
    def _write_summary(self):
        """Write final summary report"""
        with open(self.summary_file, 'w') as f:
            f.write("BATCH PROCESSING SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Started: {self.progress['started']}\n")
            f.write(f"Completed: {self.progress.get('last_updated', 'N/A')}\n")
            f.write(f"Total Runtime: {(time.time() - self.start_time)/3600:.1f} hours\n\n")
            f.write(f"Total Episodes: {self.progress['total']}\n")
            f.write(f"Successful: {self.progress['completed']}\n")
            f.write(f"Failed: {self.progress['failed']}\n\n")
            
            # Failed episodes
            failed = [(ep_id, data) for ep_id, data in self.progress['episodes'].items() 
                     if not data.get('completed', False)]
            
            if failed:
                f.write(f"FAILED EPISODES ({len(failed)}):\n")
                for ep_id, data in failed:
                    failed_stage = data.get('failed_stage', 'Unknown')
                    attempts = data.get('attempts', 0)
                    f.write(f"  - {ep_id}: Failed at Stage {failed_stage} (attempts: {attempts})\n")
                f.write("\n")
            
            # Success rate by stage
            stage_failures = {1: 0, 2: 0, 3: 0}
            for data in self.progress['episodes'].values():
                if not data.get('completed', False) and 'failed_stage' in data:
                    stage_failures[data['failed_stage']] += 1
            
            f.write("FAILURE BREAKDOWN BY STAGE:\n")
            f.write(f"  Stage 1 (WhisperX): {stage_failures[1]} failures\n")
            f.write(f"  Stage 2 (NeMo): {stage_failures[2]} failures\n")
            f.write(f"  Stage 3 (Fact Check): {stage_failures[3]} failures\n\n")
            
            f.write("Detailed results saved in individual episode directories.\n")
            f.write(f"Full log available at: {self.log_file}\n")


def get_user_input():
    """Interactive mode to get user input"""
    print("=" * 70)
    print("AUDIO DIARIZATION PIPELINE - BATCH ORCHESTRATOR")
    print("=" * 70)
    print()
    
    # Input directory
    while True:
        input_dir = input("Enter INPUT directory (containing .mp4/.wav files): ").strip()
        if not input_dir:
            print("  ❌ Input directory cannot be empty")
            continue
        
        input_path = Path(input_dir).expanduser()
        if not input_path.exists():
            print(f"  ❌ Directory not found: {input_path}")
            retry = input("  Try again? [Y/n]: ").strip().lower()
            if retry == 'n':
                sys.exit(0)
            continue
        
        # Count media files
        extensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.wav', '.mp3', '.m4a']
        media_count = sum(len(list(input_path.glob(f"**/*{ext}"))) for ext in extensions)
        
        if media_count == 0:
            print(f"  ⚠️  No media files found in {input_path}")
            retry = input("  Continue anyway? [y/N]: ").strip().lower()
            if retry != 'y':
                continue
        else:
            print(f"  ✓ Found {media_count} media files")
        
        break
    
    # Output directory
    while True:
        output_dir = input("\nEnter OUTPUT directory (for results): ").strip()
        if not output_dir:
            print("  ❌ Output directory cannot be empty")
            continue
        
        output_path = Path(output_dir).expanduser()
        
        if output_path.exists():
            # Check if resuming
            progress_file = output_path / "batch_progress.json"
            if progress_file.exists():
                print(f"  ℹ️  Found existing progress file")
                resume = input("  Resume previous run? [Y/n]: ").strip().lower()
                if resume == 'n':
                    force = True
                    print("  ⚠️  Will reprocess all episodes")
                else:
                    force = False
                    print("  ✓ Will resume from checkpoint")
            else:
                force = False
        else:
            print(f"  ℹ️  Directory will be created: {output_path}")
            force = False
        
        break
    
    # Test mode
    print("\n" + "=" * 70)
    test_mode = input("Test mode (process only a few episodes)? [y/N]: ").strip().lower()
    if test_mode == 'y':
        while True:
            max_str = input("How many episodes to test? [5]: ").strip()
            max_episodes = int(max_str) if max_str else 5
            if max_episodes > 0:
                print(f"  ✓ Will process first {max_episodes} episodes")
                break
    else:
        max_episodes = None
        print("  ✓ Will process ALL episodes")
    
    # Summary
    print("\n" + "=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"Input:        {input_path}")
    print(f"Output:       {output_path}")
    print(f"Media files:  {media_count}")
    print(f"Max episodes: {max_episodes if max_episodes else 'ALL'}")
    print(f"Force rerun:  {force}")
    print("=" * 70)
    
    confirm = input("\nStart processing? [Y/n]: ").strip().lower()
    if confirm == 'n':
        print("Aborted.")
        sys.exit(0)
    
    return str(input_path), str(output_path), max_episodes, force


def main():
    parser = argparse.ArgumentParser(
        description="Batch Orchestrator for Audio Diarization Pipeline (Server-Safe)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Modes:
  1. Interactive mode (recommended for first-time users):
     python batch_orchestrator.py
     
  2. Command-line mode:
     python batch_orchestrator.py /input /output
     
Examples:
  # Interactive mode
  python batch_orchestrator.py
  
  # Process all media files
  python batch_orchestrator.py /path/to/media_files /path/to/output
  
  # Test on first 5 episodes
  python batch_orchestrator.py /path/to/media_files /path/to/output --max 5
  
  # Force reprocess everything
  python batch_orchestrator.py /path/to/media_files /path/to/output --force

Safety Features:
  - Automatic resume after interruption
  - File locking prevents multiple instances
  - Graceful shutdown on Ctrl+C or disconnect
  - Atomic progress saves
  - Heartbeat monitoring
  - Per-stage retry logic
        """
    )
    
    parser.add_argument("input_dir", nargs='?', help="Directory containing audio/video files (.mp4, .wav, etc.)")
    parser.add_argument("output_dir", nargs='?', help="Base output directory")
    parser.add_argument("--stage1", default="/Volumes/LANDLAB/projects/Project_Sesame/ssa_sesame-street-archive/scripts/ssa_scaling/3_audio-transcriber/versions/final/stage1_whisperx.py", help="Path to Stage 1 script")
    parser.add_argument("--stage2", default="/Volumes/LANDLAB/projects/Project_Sesame/ssa_sesame-street-archive/scripts/ssa_scaling/3_audio-transcriber/versions/final/stage2_nemo.py", help="Path to Stage 2 script")
    parser.add_argument("--stage3", default="/Volumes/LANDLAB/projects/Project_Sesame/ssa_sesame-street-archive/scripts/ssa_scaling/3_audio-transcriber/versions/final/stage3_factcheck.py", help="Path to Stage 3 script")
    parser.add_argument("--force", action="store_true", help="Force reprocess all episodes")
    parser.add_argument("--max", type=int, help="Maximum number of episodes to process")
    parser.add_argument("--retry", type=int, default=3, help="Number of retry attempts per stage (default: 3)")
    parser.add_argument("--checkpoint-interval", type=int, default=1, 
                       help="Save checkpoint every N episodes (default: 1)")
    parser.add_argument("--no-heartbeat", action="store_true", help="Disable heartbeat file")
    
    args = parser.parse_args()
    
    # Interactive mode if no arguments provided
    if not args.input_dir or not args.output_dir:
        input_dir, output_dir, max_episodes, force = get_user_input()
        # Override with user input
        args.input_dir = input_dir
        args.output_dir = output_dir
        if max_episodes:
            args.max = max_episodes
        if force:
            args.force = force
    
    # Validate stage scripts exist
    stage_scripts = {
        'stage1': args.stage1,
        'stage2': args.stage2,
        'stage3': args.stage3
    }
    
    for name, path in stage_scripts.items():
        if not Path(path).exists():
            print(f"ERROR: Stage script not found: {path}")
            sys.exit(1)
    
    # Run orchestrator
    try:
        orchestrator = BatchOrchestrator(
            input_dir=args.input_dir,
            output_base_dir=args.output_dir,
            stage_scripts=stage_scripts,
            retry_failed=args.retry,
            checkpoint_interval=args.checkpoint_interval,
            enable_heartbeat=not args.no_heartbeat
        )
        
        orchestrator.run(force=args.force, max_episodes=args.max)
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user - progress has been saved")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
