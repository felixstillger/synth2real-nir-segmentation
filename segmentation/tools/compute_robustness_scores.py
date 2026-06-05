import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

def parse_model_id(model_id):
    """Attempt to extract architecture, config, and seed from model_id."""
    # Assuming model_id is like mask2former_swin-l_gta5_nir_seed0
    # For now, just use model_id as config, architecture as base, seed=0
    arch = model_id.split('_')[0] if '_' in model_id else model_id
    seed = 0
    if 'seed' in model_id:
        try:
            seed = int(model_id.split('seed')[-1])
        except:
            pass
    return arch, model_id, seed

def compute_auc_per_distortion_script(df):
    rows = []
    for (arch, cfg, dtype), sub in df.groupby(["architecture", "config", "distortion_type"], dropna=True):
        sub = sub.sort_values("intensity")
        if sub["intensity"].notna().all() and sub["intensity"].nunique() > 1:
            x = sub["intensity"].to_numpy(dtype=float)
            y = sub["mean_mIoU"].to_numpy(dtype=float)
            area = float(np.trapz(y, x))
            rng = float(x.max() - x.min())
            auc = area / rng if rng > 0 else float("nan")
            n_sev = int(sub["intensity"].nunique())
        else:
            auc = float(sub["mean_mIoU"].mean())
            n_sev = int(sub.shape[0])
        rows.append({
            "architecture": arch,
            "config": cfg,
            "distortion_type": dtype,
            "AUC": auc,
            "n_intensities": n_sev,
        })
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-csv', required=True, help='Path to results_...csv from distortion_robustness_eval.py')
    parser.add_argument('--real-baseline-config', default=None, help='Config name for real-baseline (to normalize against)')
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    df = pd.read_csv(input_path)

    # Reconstruct dr_df columns
    rows = []
    for _, row in df.iterrows():
        arch, cfg, seed = parse_model_id(row['model'])
        rows.append({
            'architecture': arch,
            'config': row['model'], # Use full model id as config for now
            'seed': seed,
            'distortion_type': row['distortion_type'],
            'intensity': row['intensity'],
            'mIoU': row.get('mIoU', np.nan),
            'aAcc': row.get('aAcc', np.nan),
            'mAcc': row.get('mAcc', np.nan),
        })
    dr_df = pd.DataFrame(rows)

    dr_agg = dr_df.groupby(
        ["architecture", "config", "distortion_type", "intensity"], dropna=False
    ).agg(
        mean_mIoU=("mIoU", "mean"),
        std_mIoU=("mIoU", "std"),
        n_seeds=("mIoU", "count"),
    ).reset_index()

    dr_auc = compute_auc_per_distortion_script(dr_agg)

    if args.real_baseline_config:
        real_ref = (
            dr_auc[
                (dr_auc["config"] == args.real_baseline_config) & dr_auc["AUC"].notna()
            ][["architecture", "distortion_type", "AUC"]]
            .rename(columns={"AUC": "AUC_ref"})
            .copy()
        )
        dr_auc_norm = dr_auc.merge(real_ref, on=["architecture", "distortion_type"], how="left")
        dr_auc_norm["AUC_norm"] = dr_auc_norm.apply(
            lambda r: 100.0 * r["AUC"] / r["AUC_ref"]
            if (pd.notna(r["AUC"]) and pd.notna(r["AUC_ref"]) and r["AUC_ref"] != 0)
            else float("nan"),
            axis=1,
        )
        dr_summary = (
            dr_auc_norm.groupby(["architecture", "config"], dropna=False)
            .agg(
                mPC_AUC=("AUC", "mean"),
                Norm_mPC_AUC=("AUC_norm", "mean"),
            )
            .reset_index()
        )
        out_summary = dr_summary
    else:
        out_summary = dr_auc.groupby(["architecture", "config"], dropna=False).agg(mPC_AUC=("AUC", "mean")).reset_index()

    out_json = input_path.parent / f"{input_path.stem}_robustness_scores.json"
    out_csv = input_path.parent / f"{input_path.stem}_robustness_scores.csv"

    out_summary.to_json(out_json, orient='records', indent=2)
    out_summary.to_csv(out_csv, index=False)

    print(f"✓ Saved robustness scores to {out_json} and {out_csv}")
    print(out_summary.to_string(index=False))

if __name__ == '__main__':
    main()
