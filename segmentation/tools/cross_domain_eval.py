#!/usr/bin/env python3
"""Cross-Domain Evaluation Script

Evaluates trained models across different domains with automatic
class set mapping and comprehensive result tracking.

Usage:
    python tools/cross_domain_eval_v2.py --config tools/cross_domain_eval_config.yaml
    python tools/cross_domain_eval_v2.py --config tools/cross_domain_eval_config.yaml --models gta5_rgb_deeplabv3plus
"""

import argparse
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml

from eval_utils import (
    build_dataset_cfg_options,
    build_metric_cfg_options,
    determine_metric_config,
    filter_dict,
    find_checkpoint,
    parse_metrics,
    save_results_json,
)


class CrossDomainEvaluator:
    """Main evaluator for cross-domain model evaluation."""
    
    def __init__(self, config_path: str, args: argparse.Namespace):
        self.args = args
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.mmseg_root = Path(__file__).parent.parent
        self.output_dir = Path(self.config['evaluation']['output_dir'])
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Filter models and datasets
        self.models = filter_dict(self.config['models'], args.models)
        self.datasets = filter_dict(self.config['test_datasets'], args.datasets)
        self.class_mappings = self.config.get('class_mappings', {})
        
        self.results = defaultdict(dict)
    
    def run_evaluation(self, model_id: str, model_cfg: Dict, 
                      dataset_id: str, dataset_cfg: Dict, checkpoint: str) -> Optional[Dict]:
        """Run evaluation and return metrics."""
        eval_dir = self.output_dir / model_id / dataset_id
        eval_dir.mkdir(exist_ok=True, parents=True)
        
        config_path = self.mmseg_root / model_cfg['config']
        
        # Build cfg-options using shared utilities
        cfg_opts = build_dataset_cfg_options(
            dataset_cfg['dataset_type'],
            dataset_cfg['data_root'],
            dataset_cfg['test_img_path'],
            dataset_cfg['test_seg_path'],
        )

        # Determine and add metric configuration
        metric_cfg = determine_metric_config(
            model_cfg.get('class_set', 'cityscapes_19'),
            dataset_cfg.get('class_set', 'cityscapes_19'),
            self.class_mappings,
        )
        cfg_opts.extend(build_metric_cfg_options(metric_cfg))
        cfg_opts.append('test_evaluator.format_only=False')
        
        # Build command
        cmd = [
            sys.executable,
            str(self.mmseg_root / 'mmsegmentation' / 'tools' / 'test.py'),
            str(config_path),
            checkpoint,
            '--work-dir', str(eval_dir),
        ]
        
        # Add visualizations if enabled (use TestSegVisualizationHook for interval support)
        vis_dir = None
        if self.config['evaluation'].get('save_visualizations', False):
            vis_dir = eval_dir / 'visualizations'
            vis_dir.mkdir(exist_ok=True)
            interval = self.config['evaluation'].get('visualization_interval', 50)
            cfg_opts.extend([
                "default_hooks.visualization.type='TestSegVisualizationHook'",
                'default_hooks.visualization.draw=True',
                f'default_hooks.visualization.interval={interval}',
            ])
            cmd.extend(['--show-dir', str(vis_dir)])
        
        # Save predictions for confusion matrix if enabled
        # Works for IoUMetric
        pred_dir = None
        if self.config['evaluation'].get('save_confusion_matrix', False):
            pred_dir = eval_dir / 'predictions'
            pred_dir.mkdir(exist_ok=True)
            cfg_opts.extend([
                f'test_evaluator.output_dir={str(pred_dir)}',
            ])
        
        cmd.extend(['--cfg-options'] + cfg_opts)
        
        # Run evaluation with W&B disabled in the spawned process. Some model
        # configs include `WandbVisBackend`, which initializes independently of
        # this wrapper script and can fail due to local artifact staging.
        child_env = os.environ.copy()
        child_env['WANDB_MODE'] = 'disabled'
        child_env['WANDB_DISABLED'] = 'true'
        subprocess.run(
            cmd,
            cwd=str(self.mmseg_root),
            env=child_env,
            capture_output=True,
            text=True,
            check=True,
        )
        
        # Parse metrics using shared utility
        metrics = parse_metrics(eval_dir)
        
        # Generate confusion matrix if predictions were saved
        if pred_dir and pred_dir.exists():
            self._generate_confusion_matrix(config_path, pred_dir, eval_dir, 
                                           model_id, dataset_id, dataset_cfg)
        
        print(f"✓ {model_id} on {dataset_id}: mIoU={metrics.get('mIoU', 0):.2f}%")
        return metrics
    
    def _generate_confusion_matrix(self, config_path: Path, pred_dir: Path,
                                   eval_dir: Path, model_id: str, 
                                   dataset_id: str, dataset_cfg: Dict):
        """Generate confusion matrix from saved predictions."""
        pred_files = list(pred_dir.glob('*.png'))
        if not pred_files:
            return
        
        print(f"  Generating confusion matrix ({len(pred_files)} predictions)...")
        
        cm_dir = eval_dir / 'confusion_matrix'
        cm_dir.mkdir(exist_ok=True)
        
        cm_script = self.mmseg_root / 'mmsegmentation' / 'tools' / 'analysis_tools' / 'confusion_matrix.py'
        cm_cmd = [
            sys.executable, str(cm_script), str(config_path),
            str(eval_dir / 'preds'), str(cm_dir),
            '--color-theme', self.config['evaluation'].get('cm_color_theme', 'winter'),
            '--title', f'{model_id} on {dataset_id}',
            '--cfg-options',
            f"test_dataloader.dataset.type={dataset_cfg['dataset_type']}",
            f"test_dataloader.dataset.data_root={dataset_cfg['data_root']}",
            f"test_dataloader.dataset.data_prefix.img_path={dataset_cfg['test_img_path']}",
            f"test_dataloader.dataset.data_prefix.seg_map_path={dataset_cfg['test_seg_path']}",
        ]
        
        child_env = os.environ.copy()
        child_env['WANDB_MODE'] = 'disabled'
        child_env['WANDB_DISABLED'] = 'true'
        subprocess.run(
            cm_cmd,
            cwd=str(self.mmseg_root),
            env=child_env,
            capture_output=True,
            text=True,
            check=True,
        )
        
        # Clean up prediction files
        shutil.rmtree(pred_dir)
        
        cm_png = cm_dir / 'confusion_matrix.png'
    
    def evaluate_all(self):
        """Run all model/dataset combinations."""
        strategy = self.config['checkpoint']['strategy']
        checkpoints = {} 
        for model_id, model_cfg in self.models.items():
            ckpt = find_checkpoint(Path(model_cfg['work_dir']), strategy)
            if ckpt:
                checkpoints[model_id] = ckpt
            else:
                print(f"Skipping {model_id}: no checkpoint found")
        
        # Run evaluations
        total = len(checkpoints) * len(self.datasets)
        current = 0
        
        for model_id, checkpoint in checkpoints.items():
            model_cfg = self.models[model_id]
            
            for dataset_id, dataset_cfg in self.datasets.items():
                current += 1
                print(f"\n[{current}/{total}] {model_cfg['name']} on {dataset_cfg['name']}")
                
                metrics = self.run_evaluation(model_id, model_cfg, dataset_id, 
                                             dataset_cfg, checkpoint)
                
                if metrics:
                    self.results[model_id][dataset_id] = metrics
        
        self.generate_summary()
    

    
    def generate_summary(self):
        """Generate and save evaluation summary."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        # Create summary data
        summary = []
        for model_id, dataset_results in self.results.items():
            model_cfg = self.models[model_id]
            for dataset_id, metrics in dataset_results.items():
                dataset_cfg = self.datasets[dataset_id]
                summary.append({
                    'model_id': model_id,
                    'model_name': model_cfg['name'],
                    'dataset_id': dataset_id,
                    'dataset_name': dataset_cfg['name'],
                    'mIoU': metrics['mIoU'],
                    'aAcc': metrics['aAcc'],
                    'mAcc': metrics['mAcc'],
                })
        
        # Save results
        df = pd.DataFrame(summary)
        
        summary_json = self.output_dir / 'summary_results.json'
        save_results_json(summary, summary_json)
        
        summary_csv = self.output_dir / 'summary_results.csv'
        df.to_csv(summary_csv, index=False)
        
        # Print table
        print("\n" + df.to_string(index=False))
        
        # Pivot table
        if not df.empty:
            pivot = df.pivot(index='model_name', columns='dataset_name', values='mIoU')
            pivot_file = self.output_dir / 'miou_pivot_table.csv'
            pivot.to_csv(pivot_file)
            print("\n" + pivot.to_string())
            
            # Domain gap analysis
            self._analyze_domain_gaps()
        

        
        print(f"\nResults saved to: {self.output_dir}")
    
    def _analyze_domain_gaps(self):
        """Calculate and save domain adaptation gaps."""
        gaps = []
        for model_id, results in self.results.items():
            model_cfg = self.models[model_id]
            train_dataset = model_cfg.get('train_dataset', '')
            
            if train_dataset in results:
                in_domain_miou = results[train_dataset]['mIoU']
                
                for test_dataset, metrics in results.items():
                    if test_dataset != train_dataset:
                        gaps.append({
                            'model': model_cfg['name'],
                            'train_dataset': train_dataset,
                            'test_dataset': test_dataset,
                            'in_domain_mIoU': in_domain_miou,
                            'cross_domain_mIoU': metrics['mIoU'],
                            'domain_gap': in_domain_miou - metrics['mIoU'],
                        })
        
        if gaps:
            gap_df = pd.DataFrame(gaps)
            gap_file = self.output_dir / 'domain_gaps.csv'
            gap_df.to_csv(gap_file, index=False)
            print("\nDomain Gap Analysis:")
            print(gap_df.to_string(index=False))
    



def main():
    parser = argparse.ArgumentParser(description='Cross-Domain Evaluation')
    parser.add_argument('--config', default='tools/cross_domain_eval_config.yaml')
    parser.add_argument('--models', nargs='+', help='Filter models by ID')
    parser.add_argument('--datasets', nargs='+', help='Filter datasets by ID')
    parser.add_argument('--skip-wandb', action='store_true', help='(no-op, wandb removed)')
    parser.add_argument('--dry-run', action='store_true', help='Print plan without running')
    args = parser.parse_args()
    
    evaluator = CrossDomainEvaluator(args.config, args)
    
    if args.dry_run:
        print(f"Would evaluate {len(evaluator.models)} models on {len(evaluator.datasets)} datasets")
        print(f"Models: {list(evaluator.models.keys())}")
        print(f"Datasets: {list(evaluator.datasets.keys())}")
        return
    
    evaluator.evaluate_all()


if __name__ == '__main__':
    main()

