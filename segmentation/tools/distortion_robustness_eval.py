"""
Distortion Robustness Evaluation

Config-driven evaluation of segmentation models on distorted image datasets.
Automatically selects the appropriate metric based on model and dataset class sets.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from eval_utils import (
    build_dataset_cfg_options,
    build_metric_cfg_options,
    determine_metric_config,
    filter_dict,
    find_checkpoint,
    parse_metrics,
)


def parse_distortion_params(dirname: str, patterns: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Parse distortion directory name using config-defined patterns."""
    for distortion_type, pattern in patterns.items():
        match = re.match(pattern, dirname)
        if match:
            params = {k: float(v) if v.replace('.', '').isdigit() else v
                      for k, v in match.groupdict().items()}
            return {'distortion_type': distortion_type, **params}
    return None


def resolve_gt_path(
    distortion: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    distortion_types: Dict[str, List[str]],
) -> str:
    """
    Determine correct GT path based on distortion type.
    
    Args:
        distortion: Distortion metadata including type and parameters
        dataset_cfg: Dataset configuration with paths
        distortion_types: Classification of geometric vs photometric distortions
    
    Returns:
        Relative path to ground truth directory (either original or distorted)
    """
    distortion_type = distortion['distortion_type']
    is_geometric = distortion_type in distortion_types.get('geometric', [])
    
    if is_geometric:
        distorted_gt_root = dataset_cfg.get('distorted_gt_root')
        if not distorted_gt_root:
            raise ValueError(
                f"distorted_gt_root not defined for {distortion['dataset_id']} "
                f"but geometric distortion {distortion_type} requires it"
            )

        # Replace image distortion root with GT distortion root, and
        # image modality prefix with GT modality prefix
        # e.g., "filtered_t10_agree80/RGB_distorted/ranus_rgb_elastic_transform_alpha50.0/test" 
        #    -> "filtered_t10_agree80/GT_8class_distorted/GT_8class_elastic_transform_alpha50.0/test"
        distortion_root = dataset_cfg['distortion_root']
        data_root = dataset_cfg.get('data_root', '')
        
        # Convert distortion_root to relative path for string replacement
        # (img_path is relative to data_root, so distortion_root must also be relative)
        try:
            relative_distortion_root = str(Path(distortion_root).relative_to(data_root))
        except ValueError:
            # Already relative or different root
            relative_distortion_root = distortion_root
        
        # Get prefix patterns from config, with fallback to legacy behavior
        img_modality_prefix = dataset_cfg.get('img_modality_prefix', 
                                               f"ranus_{distortion['modality'].lower()}")
        gt_modality_prefix = dataset_cfg.get('gt_modality_prefix', 'GT_8class')
        
        gt_path = distortion['img_path'].replace(
            relative_distortion_root, distorted_gt_root
        ).replace(img_modality_prefix, gt_modality_prefix)
    else:
        # Use original GT for photometric distortions
        gt_path = f"{dataset_cfg['gt_seg_path']}/{distortion['split']}"
    
    return gt_path


def discover_distortions(
    datasets: Dict[str, Dict],
    patterns: Dict[str, str],
    splits: List[str],
    distortion_types: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """Discover all distortion/split combinations for configured datasets."""
    distortions = []
    geometric_distortions = distortion_types.get('geometric', [])

    for dataset_id, cfg in datasets.items():
        distortion_root = Path(cfg['distortion_root'])
        data_root = Path(cfg.get('data_root', ''))
        modality = cfg.get('modality', 'rgb').upper()

        for distortion_dir in sorted(distortion_root.iterdir()):
            if not distortion_dir.is_dir():
                continue

            parsed = parse_distortion_params(distortion_dir.name, patterns)
            if not parsed:
                continue

            distortion_type = parsed['distortion_type']
            is_geometric = distortion_type in geometric_distortions

            for split in splits:
                split_dir = distortion_dir / split
                if not split_dir.exists():
                    continue

                try:
                    img_path = str(split_dir.relative_to(data_root))
                except ValueError:
                    img_path = str(split_dir)

                distortions.append({
                    'dataset_id': dataset_id,
                    'modality': modality,
                    'split': split,
                    'img_path': img_path,
                    'split_path': str(split_dir.resolve()),
                    'distortion_dir': str(distortion_dir.resolve()),
                    'is_geometric': is_geometric,
                    **parsed,
                })

    return distortions


def filter_distortions_by_paths(
    distortions: List[Dict[str, Any]],
    selected_paths: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Filter discovered distortions by exact split or distortion directory paths."""
    if not selected_paths:
        return distortions

    normalized_targets = {str(Path(path).expanduser().resolve()) for path in selected_paths}
    filtered = []

    for distortion in distortions:
        split_path = Path(distortion['split_path']).resolve()
        distortion_dir = Path(distortion['distortion_dir']).resolve()
        candidate_paths = {str(split_path), str(distortion_dir)}

        if candidate_paths & normalized_targets:
            filtered.append(distortion)

    return filtered


def make_unique_path(path: Path) -> Path:
    """Return a non-existing path by appending a numeric suffix when needed."""
    if not path.exists():
        return path

    suffix = path.suffix
    base_name = path.stem if suffix else path.name

    for idx in range(1, 10_000):
        candidate_name = f"{base_name}_{idx}{suffix}" if suffix else f"{base_name}_{idx}"
        candidate = path.parent / candidate_name
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not create a unique path for {path}")


def validate_geometric_distortions(
    datasets: Dict[str, Dict],
    distortions: List[Dict[str, Any]],
    distortion_types: Dict[str, List[str]],
) -> None:
    """Validate that geometric distortions have matching GT directories."""
    missing_gt = []
    
    for dist in distortions:
        if not dist.get('is_geometric', False):
            continue
        
        dataset_cfg = datasets[dist['dataset_id']]
        
        # Check if distorted_gt_root is configured
        if 'distorted_gt_root' not in dataset_cfg:
            raise ValueError(
                f"Missing 'distorted_gt_root' for dataset '{dist['dataset_id']}' "
                f"which has geometric distortion '{dist['distortion_type']}'"
            )
        
        # Resolve and check if GT path exists
        try:
            gt_path = resolve_gt_path(dist, dataset_cfg, distortion_types)
            full_gt_path = Path(dataset_cfg['data_root']) / gt_path
            
            if not full_gt_path.exists():
                missing_gt.append({
                    'dataset': dist['dataset_id'],
                    'distortion': dist['distortion_type'],
                    'intensity': dist.get('intensity', 'N/A'),
                    'split': dist['split'],
                    'expected_path': str(full_gt_path),
                })
        except (OSError, ValueError) as e:
            print(f"⚠ Warning: Could not validate GT path for {dist['dataset_id']} "
                  f"{dist['distortion_type']}: {e}")
    
    if missing_gt:
        print(f"\n⚠ Warning: {len(missing_gt)} geometric distortion GT directories not found:")
        for item in missing_gt[:5]:  # Show first 5
            print(f"  - {item['dataset']} {item['distortion']} (intensity={item['intensity']}, "
                  f"split={item['split']})")
            print(f"    Expected: {item['expected_path']}")
        if len(missing_gt) > 5:
            print(f"  ... and {len(missing_gt) - 5} more")
        print()


class DistortionRobustnessEvaluator:
    """Evaluator for distortion robustness experiments."""

    def __init__(self, config: Dict[str, Any], args: argparse.Namespace):
        self.config = config
        self.args = args
        self.mmseg_root = Path(__file__).parent.parent
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        eval_cfg = config['evaluation']
        self.output_dir = Path(args.output_dir or eval_cfg['output_dir']).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.splits = args.splits or eval_cfg.get('splits', ['test'])
        self.models = filter_dict(config['models'], args.models)
        self.datasets = filter_dict(config['test_datasets'], args.datasets)
        self.class_mappings = config.get('class_mappings', {})
        self.distortion_types = config.get('distortion_types', {'geometric': [], 'photometric': []})
        
        # Modality matching: if enabled, only evaluate models on datasets with matching modality
        # Can be overridden by --match-modality / --no-match-modality CLI args
        if args.match_modality is not None:
            self.match_modality = args.match_modality
        else:
            self.match_modality = eval_cfg.get('match_modality', False)

        discovered_distortions = discover_distortions(
            self.datasets,
            config['distortion_patterns'],
            self.splits,
            self.distortion_types,
        )
        self.distortions = filter_distortions_by_paths(discovered_distortions, args.distortion_paths)

        if args.distortion_paths and not self.distortions:
            raise ValueError(
                'No distortions matched the provided --distortion-paths. '
                'Pass an exact distortion directory or split directory path.'
            )

        print(f"Discovered {len(self.distortions)} distortions across {len(self.datasets)} datasets")
        
        # Count geometric vs photometric
        n_geometric = sum(1 for d in self.distortions if d.get('is_geometric', False))
        print(f"  - Geometric: {n_geometric}, Photometric: {len(self.distortions) - n_geometric}")
        
        # Validate geometric distortions have matching GT directories
        validate_geometric_distortions(self.datasets, self.distortions, self.distortion_types)

        self.results: List[Dict[str, Any]] = []


    def should_evaluate(self, model_cfg: Dict, distortion: Dict) -> bool:
        """Check if model should be evaluated on this distortion based on modality matching."""
        if not self.match_modality:
            return True
        
        model_modality = model_cfg.get('modality', '').upper()
        dataset_modality = distortion['modality'].upper()
        
        # If model modality is not specified, evaluate on all datasets
        if not model_modality:
            return True
        
        return model_modality == dataset_modality

    def build_eval_pairs(self) -> List[tuple[str, Dict[str, Any], Dict[str, Any]]]:
        """Build all model/distortion evaluation pairs after modality filtering."""
        eval_pairs = []
        for model_id, model_cfg in self.models.items():
            for distortion in self.distortions:
                if self.should_evaluate(model_cfg, distortion):
                    eval_pairs.append((model_id, model_cfg, distortion))
        return eval_pairs

    def run(self):
        """Run all evaluations."""
        strategy = self.config['checkpoint']['strategy']
        
        # Pre-filter to count actual evaluations when modality matching is enabled
        eval_pairs = self.build_eval_pairs()
        
        total = len(eval_pairs)
        if self.match_modality:
            skipped = len(self.models) * len(self.distortions) - total
            print(f"Modality matching enabled: {total} evaluations ({skipped} skipped due to modality mismatch)")
        
        idx = 0
        for model_id, model_cfg, distortion in eval_pairs:
            checkpoint = find_checkpoint(Path(model_cfg['work_dir']), strategy)
            if not checkpoint:
                print(f"⚠ Skipping {model_id}: no checkpoint found")
                continue

            idx += 1
            dataset_id = distortion['dataset_id']
            dataset_cfg = self.datasets[dataset_id]

            print(f"[{idx}/{total}] {model_id} on {dataset_id} | "
                  f"{distortion['modality']} {distortion['distortion_type']} "
                  f"(intensity={distortion['intensity']:.1f})")

            metrics = self._evaluate(model_id, model_cfg, checkpoint, dataset_cfg, distortion)
            if metrics:
                self._record(model_id, model_cfg, dataset_id, distortion, metrics)

        self._finalize()

    def _evaluate(
        self,
        model_id: str,
        model_cfg: Dict,
        checkpoint: str,
        dataset_cfg: Dict,
        distortion: Dict,
    ) -> Dict[str, float]:
        """Run a single evaluation."""
        split_output_dir = self.output_dir / model_id / distortion['dataset_id'] / \
                           f"{distortion['distortion_type']}_{distortion['intensity']}" / distortion['split']
        split_output_dir.mkdir(parents=True, exist_ok=True)
        eval_dir = make_unique_path(split_output_dir / self.run_timestamp)
        eval_dir.mkdir(parents=True, exist_ok=False)

        model_config_path = self.mmseg_root / model_cfg['config']
        config_snapshot_path = split_output_dir / model_config_path.name
        if model_config_path.exists() and not config_snapshot_path.exists():
            shutil.copy2(model_config_path, config_snapshot_path)

        # Determine metric based on class set compatibility
        model_class_set = model_cfg.get('class_set', 'cityscapes_19')
        dataset_class_set = dataset_cfg.get('class_set', 'ranus_8class')
        metric_cfg = determine_metric_config(model_class_set, dataset_class_set, self.class_mappings)

        # Resolve GT path based on distortion type (geometric vs photometric)
        gt_path = resolve_gt_path(distortion, dataset_cfg, self.distortion_types)

        # Build cfg-options using shared utilities
        cfg_opts = build_dataset_cfg_options(
            dataset_cfg['dataset_type'],
            dataset_cfg['data_root'],
            distortion['img_path'],
            gt_path,
        )
        cfg_opts.extend(build_metric_cfg_options(metric_cfg))

        cmd = [
            sys.executable,
            str(self.mmseg_root / 'tools' / 'test.py'),
            str(model_config_path),
            checkpoint,
            '--work-dir', str(eval_dir),
            '--cfg-options', *cfg_opts,
        ]

        result = subprocess.run(
            cmd,
            cwd=str(self.mmseg_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(f"✗ Evaluation failed: {result.stderr[-500:] if result.stderr else 'Unknown error'}")
            return {}

        return parse_metrics(eval_dir)

    def _record(
        self,
        model_id: str,
        model_cfg: Dict,
        dataset_id: str,
        distortion: Dict,
        metrics: Dict[str, float],
    ):
        """Record evaluation result."""
        entry = {
            'model': model_id,
            'model_name': model_cfg['name'],
            'dataset_id': dataset_id,
            'modality': distortion['modality'],
            'distortion_type': distortion['distortion_type'],
            'intensity': distortion['intensity'],
            'split': distortion['split'],
            'split_path': distortion.get('split_path', ''),
            'mIoU': metrics.get('mIoU', 0.0),
            'aAcc': metrics.get('aAcc', 0.0),
            'mAcc': metrics.get('mAcc', 0.0),
        }
        self.results.append(entry)


    def _finalize(self):
        """Save results and create plots."""
        if not self.results:
            print("⚠ No results to save")
            return

        df = pd.DataFrame(self.results)
        csv_path = make_unique_path(self.output_dir / f"results_{self.run_timestamp}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved results: {csv_path}")

        # Create plots using standalone plotting script
        plots_dir = make_unique_path(self.output_dir / f"{csv_path.stem}_plots")
        plot_script = Path(__file__).parent / 'plot_distortion_results.py'
        
        if plot_script.exists():
            try:
                subprocess.run(
                    [sys.executable, str(plot_script), str(csv_path), '--output-dir', str(plots_dir)],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"✓ Created plots in: {plots_dir}")
                
            except subprocess.CalledProcessError as e:
                print(f"⚠ Warning: Plotting failed: {e.stderr}")
        else:
            print(f"⚠ Warning: Plot script not found at {plot_script}")


def main():
    parser = argparse.ArgumentParser(description='Distortion robustness evaluation')
    parser.add_argument('--config', default='tools/distortion_robustness_config.yaml')
    parser.add_argument('--models', nargs='+', help='Filter models by ID')
    parser.add_argument('--datasets', nargs='+', help='Filter datasets by ID')
    parser.add_argument('--splits', nargs='+', help='Splits to evaluate')
    parser.add_argument('--distortion-paths', nargs='+',
                        help='Evaluate only specific distortion directories or split directories')
    parser.add_argument('--output-dir',
                        help='Override the config output directory for evaluation artifacts and summaries')
    parser.add_argument('--run-name', help='(no-op, wandb removed)')
    parser.add_argument('--offline', action='store_true', help='(no-op, wandb removed)')
    parser.add_argument('--skip-wandb', action='store_true', help='(no-op, wandb removed)')
    parser.add_argument('--dry-run', action='store_true', help='Print plan and exit')
    
    # Modality matching: mutually exclusive options
    modality_group = parser.add_mutually_exclusive_group()
    modality_group.add_argument('--match-modality', dest='match_modality', action='store_true',
                                help='Only evaluate models on datasets with matching modality (RGB on RGB, NIR on NIR)')
    modality_group.add_argument('--no-match-modality', dest='match_modality', action='store_false',
                                help='Evaluate all model-dataset combinations regardless of modality')
    parser.set_defaults(match_modality=None)  # None means use config value
    
    args = parser.parse_args()



    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    evaluator = DistortionRobustnessEvaluator(config, args)

    if args.dry_run:
        print("DRY RUN")
        print(f"Models: {list(evaluator.models.keys())}")
        print(f"Datasets: {list(evaluator.datasets.keys())}")
        print(f"Distortions: {len(evaluator.distortions)}")
        print(f"Modality matching: {evaluator.match_modality}")
        print(f"Output dir: {evaluator.output_dir}")
        if args.distortion_paths:
            print(f"Filtered distortion paths: {args.distortion_paths}")
        
        # Count actual evaluations with modality matching
        eval_count = len(evaluator.build_eval_pairs())
        total_possible = len(evaluator.models) * len(evaluator.distortions)
        print(f"Evaluations: {eval_count}/{total_possible} (modality matching: {evaluator.match_modality})")
        return

    evaluator.run()


if __name__ == '__main__':
    main()
