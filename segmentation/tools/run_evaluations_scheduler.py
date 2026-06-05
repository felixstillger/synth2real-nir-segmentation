#!/usr/bin/env python3
"""
Automated GPU Job Scheduler for NIR Evaluations

Monitors GPU availability and schedules evaluation jobs on free GPUs.
Runs in background and persists after logout.

Usage:
    python tools/run_evaluations_scheduler.py --gpus 0 1 2 3
    nohup python tools/run_evaluations_scheduler.py --gpus 0 1 2 3 > scheduler.log 2>&1 &
"""

import argparse
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False
    print("Warning: pynvml not available. Falling back to nvidia-smi parsing.")


@dataclass
class Job:
    """Represents an evaluation job."""
    job_id: str
    eval_type: str  # cross_domain, shape_bias, distortion_robustness
    architecture: str  # deeplabv3plus, segformer, mask2former
    config_path: str
    script_path: str
    priority: int  # Lower is higher priority
    
    def __str__(self):
        return f"{self.eval_type}_{self.architecture}"


class GPUMonitor:
    """Monitors GPU utilization and availability."""
    
    def __init__(self, gpu_ids: List[int], threshold: float = 50.0, 
                 stable_duration: int = 60):
        """
        Args:
            gpu_ids: List of GPU IDs to monitor
            threshold: GPU utilization threshold (%) to consider free
            stable_duration: Seconds of low utilization before considering GPU free
        """
        self.gpu_ids = gpu_ids
        self.threshold = threshold
        self.stable_duration = stable_duration
        self.low_util_start: Dict[int, Optional[float]] = {gpu: None for gpu in gpu_ids}
        
        if HAS_PYNVML:
            pynvml.nvmlInit()
            self.handles = {gpu: pynvml.nvmlDeviceGetHandleByIndex(gpu) 
                           for gpu in gpu_ids}
    
    def get_gpu_utilization(self, gpu_id: int) -> float:
        """Get current GPU utilization percentage."""
        if HAS_PYNVML:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handles[gpu_id])
                return util.gpu
            except Exception as e:
                print(f"Error reading GPU {gpu_id} via pynvml: {e}")
                return 100.0  # Assume busy on error
        else:
            # Fallback to nvidia-smi
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', 
                     '--format=csv,noheader,nounits', f'--id={gpu_id}'],
                    capture_output=True, text=True, check=True
                )
                return float(result.stdout.strip())
            except Exception as e:
                print(f"Error reading GPU {gpu_id} via nvidia-smi: {e}")
                return 100.0
    
    def get_free_gpus(self) -> List[int]:
        """Get list of GPUs that are currently free (stable low utilization)."""
        current_time = time.time()
        free_gpus = []
        
        for gpu_id in self.gpu_ids:
            util = self.get_gpu_utilization(gpu_id)
            
            if util < self.threshold:
                # GPU is below threshold
                if self.low_util_start[gpu_id] is None:
                    # Just became low - start tracking
                    self.low_util_start[gpu_id] = current_time
                else:
                    # Check if it's been stable long enough
                    duration = current_time - self.low_util_start[gpu_id]
                    if duration >= self.stable_duration:
                        free_gpus.append(gpu_id)
            else:
                # GPU is busy - reset tracking
                self.low_util_start[gpu_id] = None
        
        return free_gpus
    
    def mark_gpu_busy(self, gpu_id: int):
        """Mark GPU as busy (reset its free timer)."""
        self.low_util_start[gpu_id] = None
    
    def cleanup(self):
        """Cleanup NVML resources."""
        if HAS_PYNVML:
            pynvml.nvmlShutdown()


class JobScheduler:
    """Manages job queue and GPU assignment."""
    
    def __init__(self, gpu_monitor: GPUMonitor, log_dir: Path):
        self.gpu_monitor = gpu_monitor
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.job_queue: List[Job] = []
        self.running_jobs: Dict[int, tuple] = {}  # gpu_id -> (job, process)
        self.completed_jobs: Set[str] = set()
        self.failed_jobs: Set[str] = set()
    
    def add_job(self, job: Job):
        """Add job to queue."""
        self.job_queue.append(job)
    
    def sort_queue(self):
        """Sort queue by priority."""
        self.job_queue.sort(key=lambda j: j.priority)
    
    def start_job(self, job: Job, gpu_id: int) -> subprocess.Popen:
        """Start a job on specified GPU."""
        log_file = self.log_dir / f"{job.job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Set environment and build command
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Build command with conda environment activation
        conda_init = f"source {Path.home()}/miniconda3/etc/profile.d/conda.sh"
        conda_activate = "conda activate nir"
        python_cmd = f"python {job.script_path} --config {job.config_path} --skip-wandb"
        full_cmd = f"{conda_init} && {conda_activate} && {python_cmd}"
        
        cmd = ['bash', '-c', full_cmd]
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting {job} on GPU {gpu_id}")
        print(f"  Command: {python_cmd}")
        print(f"  Log: {log_file}")
        
        with open(log_file, 'w') as f:
            f.write(f"Job: {job}\n")
            f.write(f"GPU: {gpu_id}\n")
            f.write(f"Command: {python_cmd}\n")
            f.write(f"Conda env: nir\n")
            f.write(f"Started: {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")
            f.flush()
            
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=Path(job.script_path).parent.parent  # mmsegmentation root
            )
        
        return process
    
    def check_running_jobs(self):
        """Check status of running jobs and free up GPUs when complete."""
        completed_gpus = []
        
        for gpu_id, (job, process) in list(self.running_jobs.items()):
            poll = process.poll()
            if poll is not None:
                # Job has finished
                if poll == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ {job} completed successfully")
                    self.completed_jobs.add(job.job_id)
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ {job} failed with code {poll}")
                    self.failed_jobs.add(job.job_id)
                
                completed_gpus.append(gpu_id)
        
        # Remove completed jobs and mark GPUs as potentially free
        for gpu_id in completed_gpus:
            del self.running_jobs[gpu_id]
    
    def schedule(self):
        """Main scheduling loop - assign jobs to free GPUs."""
        self.check_running_jobs()
        
        # Get currently free GPUs
        free_gpus = self.gpu_monitor.get_free_gpus()
        # Exclude GPUs with running jobs
        free_gpus = [gpu for gpu in free_gpus if gpu not in self.running_jobs]
        
        if not free_gpus or not self.job_queue:
            return
        
        # Assign jobs to free GPUs
        for gpu_id in free_gpus:
            if not self.job_queue:
                break
            
            job = self.job_queue.pop(0)
            process = self.start_job(job, gpu_id)
            self.running_jobs[gpu_id] = (job, process)
            self.gpu_monitor.mark_gpu_busy(gpu_id)
    
    def is_complete(self) -> bool:
        """Check if all jobs are done."""
        return len(self.job_queue) == 0 and len(self.running_jobs) == 0
    
    def print_status(self):
        """Print current status."""
        print(f"\n{'='*80}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Status:")
        print(f"  Queued: {len(self.job_queue)}")
        print(f"  Running: {len(self.running_jobs)}")
        print(f"  Completed: {len(self.completed_jobs)}")
        print(f"  Failed: {len(self.failed_jobs)}")
        
        if self.running_jobs:
            print(f"\n  Currently running:")
            for gpu_id, (job, _) in self.running_jobs.items():
                print(f"    GPU {gpu_id}: {job}")
        
        if self.job_queue:
            print(f"\n  Next in queue:")
            for i, job in enumerate(self.job_queue[:3]):
                print(f"    {i+1}. {job}")
            if len(self.job_queue) > 3:
                print(f"    ... and {len(self.job_queue) - 3} more")
        
        print(f"{'='*80}\n")


def create_job_queue() -> List[Job]:
    """Create prioritized job queue."""
    mmseg_root = Path(__file__).parent.parent
    tools_dir = mmseg_root / "tools"
    
    # Define evaluation types with priorities and scripts
    eval_types = [
        ("cross_domain", "cross_domain_eval.py", 0),  # Fastest
        ("shape_bias", "shape_bias_eval.py", 100),       # Medium
        ("distortion_robustness", "distortion_robustness_eval.py", 200),  # Slowest
    ]
    
    # Define architectures in priority order
    architectures = [
        ("deeplabv3plus", 0),   # Shortest
        ("segformer", 10),      # Medium
        ("mask2former", 20),    # Longest
    ]
    
    jobs = []
    
    for eval_type, script_name, eval_priority in eval_types:
        for arch, arch_priority in architectures:
            # Construct config path based on eval type
            if eval_type == "cross_domain":
                config_name = f"cross_domain_eval_config_{arch}.yaml"
            elif eval_type == "shape_bias":
                config_name = f"shape_bias_config_{arch}.yaml"
            else:  # distortion_robustness
                config_name = f"distortion_robustness_config_{arch}_simplified.yaml"
            
            config_path = tools_dir / config_name
            script_path = tools_dir / script_name
            
            # Check if files exist
            if not config_path.exists():
                print(f"Warning: Config not found: {config_path}")
                continue
            if not script_path.exists():
                print(f"Warning: Script not found: {script_path}")
                continue
            
            job_id = f"{eval_type}_{arch}"
            priority = eval_priority + arch_priority
            
            jobs.append(Job(
                job_id=job_id,
                eval_type=eval_type,
                architecture=arch,
                config_path=str(config_path),
                script_path=str(script_path),
                priority=priority
            ))
    
    # Sort by priority
    jobs.sort(key=lambda j: j.priority)
    
    return jobs


def main():
    parser = argparse.ArgumentParser(
        description='Automated GPU Job Scheduler for NIR Evaluations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor GPUs 0-3 and run evaluations:
  python tools/run_evaluations_scheduler.py --gpus 0 1 2 3
  
  # Run in background:
  nohup python tools/run_evaluations_scheduler.py --gpus 0 1 2 3 > scheduler.log 2>&1 &
  
  # Monitor with custom thresholds:
  python tools/run_evaluations_scheduler.py --gpus 0 1 --threshold 40 --stable-time 120
        """
    )
    parser.add_argument('--gpus', type=int, nargs='+', required=True,
                       help='GPU IDs to monitor and use')
    parser.add_argument('--threshold', type=float, default=40.0,
                       help='GPU utilization threshold (%%) to consider free (default: 40)')
    parser.add_argument('--stable-time', type=int, default=120,
                       help='Seconds of low utilization before considering GPU free (default: 120)')
    parser.add_argument('--check-interval', type=int, default=30,
                       help='Seconds between status checks (default: 30)')
    parser.add_argument('--log-dir', type=str, default='work_dirs/scheduler_logs',
                       help='Directory for job logs (default: work_dirs/scheduler_logs)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print job queue without running')
    
    args = parser.parse_args()
    
    # Create job queue
    print("Creating job queue...")
    jobs = create_job_queue()
    
    if not jobs:
        print("Error: No valid jobs found!")
        return 1
    
    print(f"\nTotal jobs: {len(jobs)}")
    print("\nJob queue (in priority order):")
    for i, job in enumerate(jobs, 1):
        print(f"  {i:2d}. [{job.priority:3d}] {job}")
    
    if args.dry_run:
        print("\nDry run - exiting without execution.")
        return 0
    
    # Initialize GPU monitor and scheduler
    print(f"\nInitializing GPU monitor for GPUs: {args.gpus}")
    print(f"  Threshold: {args.threshold}% utilization")
    print(f"  Stable duration: {args.stable_time}s")
    
    gpu_monitor = GPUMonitor(
        gpu_ids=args.gpus,
        threshold=args.threshold,
        stable_duration=args.stable_time
    )
    
    log_dir = Path(args.log_dir)
    scheduler = JobScheduler(gpu_monitor, log_dir)
    
    # Add all jobs to scheduler
    for job in jobs:
        scheduler.add_job(job)
    
    print(f"\nStarting scheduler (logs: {log_dir})")
    print("Press Ctrl+C to stop (running jobs will continue)\n")
    
    try:
        status_counter = 0
        while not scheduler.is_complete():
            scheduler.schedule()
            
            # Print status periodically
            status_counter += 1
            if status_counter >= 10:  # Every 10 iterations
                scheduler.print_status()
                status_counter = 0
            
            time.sleep(args.check_interval)
        
        # Final status
        print("\n" + "="*80)
        print("ALL JOBS COMPLETE!")
        scheduler.print_status()
        
        if scheduler.failed_jobs:
            print("\nFailed jobs:")
            for job_id in scheduler.failed_jobs:
                print(f"  - {job_id}")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nScheduler interrupted by user.")
        print(f"Running jobs will continue (PIDs in logs: {log_dir})")
        scheduler.print_status()
        return 0
    
    finally:
        gpu_monitor.cleanup()


if __name__ == '__main__':
    sys.exit(main())
