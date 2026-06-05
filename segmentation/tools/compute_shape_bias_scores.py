import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

def parse_model_id(model_id):
    arch = model_id.split('_')[0] if '_' in model_id else model_id
    seed = 0
    if 'seed' in model_id:
        try:
            seed = int(model_id.split('seed')[-1])
        except:
            pass
    return arch, model_id, seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape-bias-csv', required=True, help='Path to detailed_results.csv from shape_bias_eval.py')
    parser.add_argument('--cross-domain-csv', required=True, help='Path to summary_results.csv from cross_domain_eval.py')
    parser.add_argument('--clean-dataset-id', default='ranus_nir', help='Dataset ID for QO (clean cross-domain)')
    args = parser.parse_args()

    sb_df = pd.read_csv(args.shape_bias_csv)
    cd_df = pd.read_csv(args.cross_domain_csv)

    VOR_KEYS = ["voronoi_64", "voronoi_128"]  # Using dataset names from eval
    EED_KEY = "eed"

    sb_results = {}
    for _, row in sb_df.iterrows():
        arch, cfg, seed = parse_model_id(row['model_id'])
        ds = row['dataset_id'].lower()
        key = None
        if 'eed' in ds:
            key = EED_KEY
        elif 'voronoi_64' in ds:
            key = "voronoi_64"
        elif 'voronoi_128' in ds:
            key = "voronoi_128"
            
        if key:
            sb_results.setdefault(arch, {}).setdefault(cfg, {}).setdefault(seed, {})[key] = {'mIoU': row.get('miou', row.get('mIoU', np.nan))}

    cd_results = {}
    for _, row in cd_df.iterrows():
        arch, cfg, seed = parse_model_id(row['model_id'])
        if args.clean_dataset_id.lower() in row['dataset_id'].lower():
            cd_results.setdefault(arch, {}).setdefault(cfg, {})[seed] = {'mIoU': row.get('mIoU', np.nan)}

    def mean_over_seeds(arch, cfg, dist_key):
        seeds_data = sb_results.get(arch, {}).get(cfg, {})
        vals = [d.get(dist_key, {}).get('mIoU') for d in seeds_data.values()]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        return float(np.mean(vals)) if vals else None

    def cd_mean_over_seeds(arch, cfg):
        seeds_data = cd_results.get(arch, {}).get(cfg, {})
        vals = [d.get('mIoU') for d in seeds_data.values()]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        return float(np.mean(vals)) if vals else None

    scores = {}
    out_records = []
    
    # Get all active archs and configs
    all_archs = set(list(sb_results.keys()) + list(cd_results.keys()))
    
    for vk in VOR_KEYS:
        entries = []
        for arch in all_archs:
            cfgs = set(list(sb_results.get(arch, {}).keys()) + list(cd_results.get(arch, {}).keys()))
            for cfg in cfgs:
                QS = mean_over_seeds(arch, cfg, EED_KEY)
                QT = mean_over_seeds(arch, cfg, vk)
                QO = cd_mean_over_seeds(arch, cfg)
                entries.append({"arch": arch, "cfg": cfg, "QS": QS, "QT": QT, "QO": QO})

        QS_all = [e["QS"] for e in entries if e["QS"] is not None]
        QT_all = [e["QT"] for e in entries if e["QT"] is not None]
        s = float(np.mean(QS_all)) if QS_all else float("nan")
        t = float(np.mean(QT_all)) if QT_all else float("nan")

        scores[vk] = {}
        for e in entries:
            QS, QT, QO = e["QS"], e["QT"], e["QO"]
            SB = QS / (QS + QT) if (QS is not None and QT is not None and (QS + QT) > 0) else None

            Scd = None
            if QS is not None and QT is not None and not math.isnan(s) and not math.isnan(t) and s > 0 and t > 0:
                num = QS / s
                den = num + (QT / t)
                Scd = float(num / den) if den > 0 else None

            Rcd = None
            if QS is not None and QT is not None and QO is not None and QO > 0:
                Rcd = (QS + QT) / (2.0 * QO)

            scores[vk].setdefault(e["arch"], {})[e["cfg"]] = {
                "QS": QS, "QT": QT, "QO": QO, "SB": SB, "Scd": Scd, "Rcd": Rcd,
            }
            out_records.append({
                "voronoi_key": vk, "architecture": e["arch"], "config": e["cfg"],
                "QS": QS, "QT": QT, "QO": QO, "SB": SB, "Scd": Scd, "Rcd": Rcd
            })

    out_df = pd.DataFrame(out_records)
    out_json = Path(args.shape_bias_csv).parent / "shape_bias_postprocessed_scores.json"
    out_csv = Path(args.shape_bias_csv).parent / "shape_bias_postprocessed_scores.csv"

    out_df.to_json(out_json, orient='records', indent=2)
    out_df.to_csv(out_csv, index=False)
    
    print(f"✓ Saved shape bias post-processed scores to {out_json} and {out_csv}")
    print(out_df.dropna(subset=['Scd', 'Rcd']).to_string(index=False))

if __name__ == '__main__':
    main()
