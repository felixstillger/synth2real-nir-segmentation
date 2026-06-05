"""Shared utilities for evaluation scripts.

Common functionality for cross-domain and distortion robustness evaluation.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def determine_metric_config(
    model_class_set: str,
    dataset_class_set: str,
    class_mappings: dict,
) -> dict:
    return {'type': 'IoUMetric'}
    
    # Look for mapping model->dataset
    mapping_key = f"{model_class_set}_to_{dataset_class_set}"
    if mapping_key in class_mappings:
        mapping = class_mappings[mapping_key]
        return {
            'type': 'CrossDomainIoUMetric',
            'pred_mapping': mapping.get('pred_mapping', 'none'),
            'gt_mapping': mapping.get('gt_mapping', 'none'),
        }
    
    # Check reverse mapping (map GT instead of predictions)
    reverse_key = f"{dataset_class_set}_to_{model_class_set}"
    if reverse_key in class_mappings:
        mapping = class_mappings[reverse_key]
        return {
            'type': 'CrossDomainIoUMetric',
            'pred_mapping': 'none',
            'gt_mapping': mapping.get('pred_mapping', 'none'),
        }
    
    return {'type': 'IoUMetric'}


def find_checkpoint(work_dir: Path, strategy: str = 'best') -> Optional[str]:
    """Find checkpoint based on strategy or explicit checkpoint name.
    
    Args:
        work_dir: Directory containing checkpoints
        strategy: 'best' for best_mIoU checkpoint, 'latest' for highest
            iteration, or a specific checkpoint name/path such as
            'iter_30000' or 'iter_30000.pth'
        
    Returns:
        Path to checkpoint file or None if not found
    """
    if not work_dir.exists():
        return None

    if strategy not in {'best', 'latest'}:
        checkpoint_path = Path(strategy)
        if checkpoint_path.suffix != '.pth':
            checkpoint_path = checkpoint_path.with_suffix('.pth')

        if not checkpoint_path.is_absolute():
            checkpoint_path = work_dir / checkpoint_path

        if checkpoint_path.exists():
            return str(checkpoint_path)

        return None
    
    if strategy == 'best':
        ckpts = sorted(work_dir.glob('best_mIoU_*.pth'), key=lambda p: p.stat().st_mtime)
        if ckpts:
            return str(ckpts[-1])
    
    # Fallback to latest iter_* or epoch_*
    for pattern in ['iter_*.pth', 'epoch_*.pth']:
        ckpts = list(work_dir.glob(pattern))
        if ckpts:
            ckpts.sort(key=lambda p: int(p.stem.split('_')[1]))
            return str(ckpts[-1])
    
    return None


def parse_metrics(work_dir: Path) -> Dict[str, float]:
    """Parse metrics from mmseg evaluation output files.
    
    Args:
        work_dir: Directory containing metrics.json or scalars.json
        
    Returns:
        Dict with mIoU, aAcc, mAcc and any other numeric metrics
    """
    for filename in ['metrics.json', 'scalars.json']:
        metrics_file = work_dir / filename
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
            return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
    return {}


def build_metric_cfg_options(metric_cfg: dict) -> list:
    return [f"test_evaluator.type={metric_cfg['type']}"]


def build_dataset_cfg_options(
    dataset_type: str,
    data_root: str,
    img_path: str,
    seg_path: str,
) -> List[str]:
    """Build cfg-options list for dataset configuration.
    
    Args:
        dataset_type: MMSeg dataset class name
        data_root: Root directory for dataset
        img_path: Path to images relative to data_root
        seg_path: Path to segmentation maps relative to data_root
        
    Returns:
        List of cfg-option strings for mmseg test.py
    """
    return [
        f"test_dataloader.dataset.type={dataset_type}",
        f"test_dataloader.dataset.data_root={data_root}",
        f"test_dataloader.dataset.data_prefix.img_path={img_path}",
        f"test_dataloader.dataset.data_prefix.seg_map_path={seg_path}",
    ]


def run_mmseg_test(
    mmseg_root: Path,
    config_path: Path,
    checkpoint: str,
    work_dir: Path,
    cfg_options: List[str],
    show_dir: Optional[Path] = None,
) -> bool:
    """Run mmseg test.py with specified configuration.
    
    Args:
        mmseg_root: Root directory of mmsegmentation
        config_path: Path to model config file
        checkpoint: Path to checkpoint file
        work_dir: Output directory for results
        cfg_options: List of cfg-option strings
        show_dir: Optional directory for visualization output
        
    Returns:
        True if evaluation succeeded, False otherwise
    """
    cmd = [
        sys.executable,
        str(mmseg_root / 'tools' / 'test.py'),
        str(config_path),
        checkpoint,
        '--work-dir', str(work_dir),
        '--cfg-options', *cfg_options,
    ]
    
    if show_dir:
        cmd.extend(['--show-dir', str(show_dir)])
    
    result = subprocess.run(cmd, cwd=str(mmseg_root), capture_output=True, text=True)
    return result.returncode == 0


def filter_dict(items: Dict, keys: Optional[List[str]]) -> Dict:
    """Filter dictionary by keys if provided.
    
    Args:
        items: Dictionary to filter
        keys: List of keys to keep, or None to keep all
        
    Returns:
        Filtered dictionary
    """
    if keys:
        return {k: v for k, v in items.items() if k in keys}
    return items


def save_results_csv(results: List[Dict], output_path: Path) -> pd.DataFrame:
    """Save results list to CSV and return DataFrame.
    
    Args:
        results: List of result dictionaries
        output_path: Path for CSV file
        
    Returns:
        DataFrame of results
    """
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    return df


def save_results_json(results: List[Dict], output_path: Path) -> None:
    """Save results list to JSON file.
    
    Args:
        results: List of result dictionaries
        output_path: Path for JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
