import sys
import os
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "adv_toys")

RUN_SPECS = [
    # 3DGenScore runs
    ("MEt3R", "met3r_run", "met3r_checkpoint.json", "met3r", "score", True),
    ("3DGenScore", "3dgenscore_run", "3dgenscore_checkpoint.json", "3dgenscore", "geometry_score", False),  # geom
    ("Hi3DEval", "hi3deval_run", "hi3deval_checkpoint.json", "hi3deval", "geometry_score", False),
    ("M3aSURE-Zero", "projector_run", "projector_checkpoint.json", "projector", "long_range_score", False),
    ("M3aSURE-One", "latent_alignment_run", "latent_alignment_checkpoint.json", "latent_alignment", "training_score",
     False),
]


def analyze_adv_toys4k(run_specs, save_csv=None):
    results = []

    for name, run_id, filename, base_key, score_key, invert in run_specs:
        file_path = os.path.join(RESULTS_ROOT, run_id, filename)
        if not os.path.isfile(file_path):
            print(f"Warning: {file_path} not found. Skipping {name}.")
            continue

        with open(file_path, "r") as f:
            data = json.load(f)

        deltas = []
        clean_scores = []
        poisoned_scores = []

        for battle_id, item in data.items():
            key_without = f"{base_key}_without_oddity"
            key_with = f"{base_key}_with_oddity"

            if key_without not in item or key_with not in item:
                continue

            w_score = item[key_without].get(score_key)
            o_score = item[key_with].get(score_key)

            if w_score is not None and o_score is not None:
                diff = w_score - o_score
                if invert:
                    diff = -diff
                deltas.append(diff)
                clean_scores.append(w_score)
                poisoned_scores.append(o_score)

        if not deltas:
            print(f"Warning: No valid scores found for {name}.")
            continue

        arr_deltas = np.array(deltas)
        arr_clean = np.array(clean_scores)
        arr_poisoned = np.array(poisoned_scores)

        results.append({
            "Model": name,
            # "n_trials": len(arr_deltas),
            # "delta_mean": np.mean(arr_deltas),
            # "delta_std": np.std(arr_deltas),
            # "clean_mean": np.mean(arr_clean),
            # "poisoned_mean": np.mean(arr_poisoned),
            "Acc. $\\uparrow$": np.mean(arr_deltas > 0),
        })

    df = pd.DataFrame(results)

    if save_csv and not df.empty:
        df.to_csv(save_csv, index=False)
        print(f"Saved results to {save_csv}")

    return df


def format_latex_table_cvpr_custom(df: pd.DataFrame) -> str:
    name_mapping = {
        "MEt3R": r"MEt3R \cite{asim_met3r_2026}",
        "3DGenScore": r"3DGenScore \cite{zhang_3dgen-bench_2025}",
        "Hi3DEval": r"Hi3DEval \cite{zhang_hi3deval_2025}",
        "M3aSURE-Zero": r"\textbf{M3aSURE-Zero}",
        "M3aSURE-One": r"\textbf{M3aSURE-One}",
    }

    our_metrics = ["M3aSURE-Zero", "M3aSURE-One"]
    cols_to_max = ["Acc. $\\uparrow$"]

    max_vals = {}
    for col in cols_to_max:
        max_vals[col] = df[col].astype(float).max()

    lines = []
    lines.append(r"\begin{tabular}{l c}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Metric} & \textbf{Acc. $\uparrow$} \\")
    lines.append(r"    \midrule")

    has_rendered_ours = False

    for _, row in df.iterrows():
        metric = row["Model"]

        if metric in our_metrics and not has_rendered_ours:
            lines.append(r"    \midrule")
            has_rendered_ours = True

        metric_name = name_mapping.get(metric, metric)
        formatted_row = [metric_name]

        for col in cols_to_max:
            val_float = float(row[col])
            val_str = f"{val_float:.3f}"

            if abs(val_float - max_vals[col]) < 1e-6:
                formatted_row.append(f"\\textbf{{{val_str}}}")
            else:
                formatted_row.append(val_str)

        lines.append("    " + " & ".join(formatted_row) + r" \\")

    lines.append(r"    \bottomrule")
    lines.append(r"\end{tabular}")

    return "\n".join(lines)


if __name__ == "__main__":
    df = analyze_adv_toys4k(
        RUN_SPECS,
        # save_csv="../results/advtoys4k_robustness.csv"
    )

    print(format_latex_table_cvpr_custom(df))
