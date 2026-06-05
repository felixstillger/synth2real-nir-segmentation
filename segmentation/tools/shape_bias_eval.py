"""
Shape & Texture Bias Evaluation

Config-driven evaluation of segmentation models on shape-biased (EED) and 
texture-biased (Voronoi) benchmark datasets.

Calculates shape and texture bias scores:
- shape_bias = eed_miou / (eed_miou + voronoi_miou)
- texture_bias = voronoi_miou / (eed_miou + voronoi_miou)
"""

import argparse
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml


# Add tools directory to path for imports when running from mmsegmentation root
import os
import sys
tools_dir = os.path.dirname(os.path.abspath(__file__))
if tools_dir not in sys.path:
    sys.path.insert(0, tools_dir)


from eval_utils import (
    build_dataset_cfg_options,
    build_metric_cfg_options,
    determine_metric_config,
    filter_dict,
    find_checkpoint,
    parse_metrics,
    run_mmseg_test,
)


class ShapeBiasEvaluator:
    """Evaluator for shape/texture bias benchmarks."""

    def __init__(self, config: Dict[str, Any], args: argparse.Namespace):
        self.config = config
        self.args = args
        self.mmseg_root = Path(__file__).parent.parent

        eval_cfg = config['evaluation']
        self.output_dir = Path(eval_cfg['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.splits = args.splits or eval_cfg.get('splits', ['test'])
        self.models = filter_dict(config['models'], args.models)
        self.benchmark_datasets = filter_dict(config['benchmark_datasets'], args.datasets)
        self.benchmark_pairs = config['benchmark_pairs']
        self.class_mappings = config.get('class_mappings', {})

        print(f"Configured {len(self.models)} models")
        print(f"Configured {len(self.benchmark_datasets)} benchmark datasets")
        print(f"Configured {len(self.benchmark_pairs)} benchmark pairs")

        self.results: List[Dict[str, Any]] = []

    def run(self):
        """Run all evaluations and compute shape/texture bias scores."""
        strategy = self.config['checkpoint']['strategy']
        total = len(self.models) * len(self.benchmark_datasets) * len(self.splits)
        idx = 0

        # Step 1: Run inference on all benchmark datasets
        for model_id, model_cfg in self.models.items():
            checkpoint = find_checkpoint(Path(model_cfg['work_dir']), strategy)
            if not checkpoint:
                print(f"⚠ Checkpoint not found for {model_id}, skipping")
                continue

            print(f"\n{'='*80}")
            print(f"Model: {model_cfg['name']}")
            print(f"Checkpoint: {checkpoint}")
            print(f"{'='*80}")

            for dataset_id, dataset_cfg in self.benchmark_datasets.items():
                for split in self.splits:
                    idx += 1
                    print(f"\n[{idx}/{total}] Evaluating {model_id} on {dataset_id}/{split}")
                    
                    metrics = self._evaluate(
                        model_id, model_cfg, checkpoint,
                        dataset_id, dataset_cfg, split
                    )
                    
                    self._record(model_id, model_cfg, dataset_id, dataset_cfg, split, metrics)

        # Step 2: Calculate shape and texture bias scores
        self._calculate_bias_scores()

        # Step 3: Finalize and save results
        self._finalize()

    def _evaluate(
        self,
        model_id: str,
        model_cfg: Dict,
        checkpoint: str,
        dataset_id: str,
        dataset_cfg: Dict,
        split: str,
    ) -> Dict[str, float]:
        """Run inference on a single benchmark dataset."""
        eval_dir = self.output_dir / model_id / dataset_id / split
        eval_dir.mkdir(parents=True, exist_ok=True)

        # Determine metric based on class set compatibility
        model_class_set = model_cfg.get('class_set', 'cityscapes_19')
        dataset_class_set = dataset_cfg.get('class_set', 'ranus_8class')
        metric_cfg = determine_metric_config(model_class_set, dataset_class_set, self.class_mappings)

        # Use paths directly from config (complete paths with split included)
        img_path = dataset_cfg['img_path']
        seg_path = dataset_cfg['seg_path']
        
        # Build cfg-options using shared utilities
        cfg_opts = build_dataset_cfg_options(
            dataset_cfg['dataset_type'],
            dataset_cfg['data_root'],
            img_path,
            seg_path,
        )
        cfg_opts.extend(build_metric_cfg_options(metric_cfg))

        # Run mmseg test
        run_mmseg_test(
            self.mmseg_root,
            Path(model_cfg['config']),
            checkpoint,
            eval_dir,
            cfg_opts,
        )

        # Parse and return metrics
        return parse_metrics(eval_dir)

    def _record(
        self,
        model_id: str,
        model_cfg: Dict,
        dataset_id: str,
        dataset_cfg: Dict,
        split: str,
        metrics: Dict[str, float],
    ):
        """Store evaluation result."""
        result = {
            'model_id': model_id,
            'model_name': model_cfg['name'],
            'model_class_set': model_cfg.get('class_set', 'unknown'),
            'model_modality': model_cfg.get('modality', 'unknown'),
            'dataset_id': dataset_id,
            'dataset_name': dataset_cfg['name'],
            'dataset_class_set': dataset_cfg.get('class_set', 'unknown'),
            'benchmark_type': dataset_cfg.get('benchmark_type'),  # 'shape' | 'texture' | None
            'split': split,
            'miou': metrics.get('mIoU'),
            'macc': metrics.get('mAcc'),
            'aacc': metrics.get('aAcc'),
        }
        self.results.append(result)


    def _calculate_bias_scores(self):
        """Calculate shape and texture bias scores for all models."""
        df = pd.DataFrame(self.results)
        scores = []

        # Optional normalization constants (s/t) for expert scaling
        norm_cfg = self.config.get('normalization_constants', {})

        for model_id in df['model_id'].unique():
            model_results = df[df['model_id'] == model_id]
            model_cfg = self.models[model_id]

            # Calculate score for each benchmark pair
            for pair in self.benchmark_pairs:
                # Find shape (EED) and texture (Voronoi) results
                shape_results = model_results[
                    model_results['dataset_id'].str.contains(pair['shape_dataset'])
                ]
                texture_results = model_results[
                    model_results['dataset_id'].str.contains(pair['texture_dataset'])
                ]

                if shape_results.empty or texture_results.empty:
                    print(f"⚠ Missing results for {model_id} on {pair['name']}")
                    continue

                # Average across splits if multiple
                shape_miou = shape_results['miou'].mean()
                texture_miou = texture_results['miou'].mean()

                # Optional normalized bias (requires s/t constants); default to simple ratio
                use_norm = pair.get('use_normalization', False)
                norm_domain = model_cfg.get('norm_domain') or model_cfg.get('train_domain') or model_cfg.get('class_set')
                norm_constants = norm_cfg.get(norm_domain, {}) if use_norm else None

                if use_norm and norm_constants:
                    s_key = pair.get('norm_shape_key')
                    t_key = pair.get('norm_texture_key')
                    s = norm_constants.get(s_key) if s_key else None
                    t = norm_constants.get(t_key) if t_key else None

                    if s and t and s > 0 and t > 0:
                        shape_scaled = shape_miou / s
                        texture_scaled = texture_miou / t
                        total_perf = shape_scaled + texture_scaled
                        shape_bias = shape_scaled / total_perf if total_perf > 0 else 0
                        texture_bias = texture_scaled / total_perf if total_perf > 0 else 0
                    else:
                        print(f"⚠ Normalization constants missing for {model_id} ({norm_domain}); falling back to simple ratio")
                        total_perf = shape_miou + texture_miou
                        shape_bias = shape_miou / total_perf if total_perf > 0 else 0
                        texture_bias = texture_miou / total_perf if total_perf > 0 else 0
                else:
                    total_perf = shape_miou + texture_miou
                    shape_bias = shape_miou / total_perf if total_perf > 0 else 0
                    texture_bias = texture_miou / total_perf if total_perf > 0 else 0

                scores.append({
                    'model_id': model_id,
                    'model_name': model_cfg['name'],
                    'model_modality': model_cfg.get('modality', 'unknown'),
                    'benchmark_pair': pair['name'],
                    'shape_miou': shape_miou,
                    'texture_miou': texture_miou,
                    'shape_bias': shape_bias,
                    'texture_bias': texture_bias,
                })


        self.scores_df = pd.DataFrame(scores)

    def _finalize(self):
        """Save results and generate outputs."""
        # Save detailed evaluation results
        results_csv = self.output_dir / "detailed_results.csv"
        pd.DataFrame(self.results).to_csv(results_csv, index=False)
        print(f"\nDetailed results saved to: {results_csv}")

        # Save shape/texture bias scores
        scores_csv = self.output_dir / "bias_scores.csv"
        self.scores_df.to_csv(scores_csv, index=False)
        print(f"Bias scores saved to: {scores_csv}")

        # Save pivot tables for shape bias
        shape_pivot = self.scores_df.pivot_table(
            index='model_name',
            columns='benchmark_pair',
            values='shape_bias',
        )
        shape_pivot_csv = self.output_dir / "shape_bias_pivot.csv"
        shape_pivot.to_csv(shape_pivot_csv)
        print(f"Shape bias pivot saved to: {shape_pivot_csv}")

        # Save pivot tables for texture bias
        texture_pivot = self.scores_df.pivot_table(
            index='model_name',
            columns='benchmark_pair',
            values='texture_bias',
        )
        texture_pivot_csv = self.output_dir / "texture_bias_pivot.csv"
        texture_pivot.to_csv(texture_pivot_csv)
        print(f"Texture bias pivot saved to: {texture_pivot_csv}")

        # Save JSON
        scores_json = self.output_dir / "bias_scores.json"
        self.scores_df.to_json(scores_json, orient='records', indent=2)


        # Print summary
        print(f"\n{'='*80}")
        print(f"Shape & Texture Bias Evaluation Complete")
        print(f"{'='*80}")
        print(f"\nResults saved to: {self.output_dir}")
        print(f"\nShape Bias Scores:")
        print(shape_pivot.to_string())

        print(f"\nTexture Bias Scores:")
        print(texture_pivot.to_string())



def main():
    parser = argparse.ArgumentParser(description='Shape & texture bias evaluation')
    parser.add_argument('--config', default='tools/shape_bias_config.yaml', help='Path to config file')
    parser.add_argument('--models', nargs='+', help='Filter models by ID')
    parser.add_argument('--datasets', nargs='+', help='Filter datasets by ID')
    parser.add_argument('--splits', nargs='+', help='Splits to evaluate (default: from config)')
    parser.add_argument('--run-name', help='(no-op, wandb removed)')
    parser.add_argument('--offline', action='store_true', help='(no-op, wandb removed)')
    parser.add_argument('--skip-wandb', action='store_true', help='(no-op, wandb removed)')
    parser.add_argument('--dry-run', action='store_true', help='Print plan and exit')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    evaluator = ShapeBiasEvaluator(config, args)

    if args.dry_run:
        print(f"\n{'='*80}")
        print("Shape Bias Evaluation Plan (Dry Run)")
        print(f"{'='*80}")
        print(f"\nModels ({len(evaluator.models)}):")
        for model_id, cfg in evaluator.models.items():
            print(f"  - {model_id}: {cfg['name']}")
        print(f"\nBenchmark Datasets ({len(evaluator.benchmark_datasets)}):")
        for ds_id, cfg in evaluator.benchmark_datasets.items():
            print(f"  - {ds_id}: {cfg['name']} ({cfg.get('benchmark_type', 'unknown')})")
        print(f"\nBenchmark Pairs ({len(evaluator.benchmark_pairs)}):")
        for pair in evaluator.benchmark_pairs:
            print(f"  - {pair['name']}: {pair['shape_dataset']} vs {pair['texture_dataset']}")
        print(f"\nSplits: {evaluator.splits}")
        print(f"\nTotal evaluations: {len(evaluator.models) * len(evaluator.benchmark_datasets) * len(evaluator.splits)}")
        return

    evaluator.run()


if __name__ == '__main__':
    main()
