from tabulate import tabulate
import json
from typing import Callable
import numpy as np
from sklearn.metrics import roc_auc_score
import os
import json
import pandas as pd


def _battle_outcome(a, b):
    """Return ordinal class in {0, 1, 2} from A's perspective: 2 if A>B, 1 if tie, 0 if A<B."""
    if a > b:
        return 2
    if a < b:
        return 0
    return 1


def compute_3dgen_metrics_continuous(
        scores: dict,
        battles_data: dict,
        criteria: tuple[str, ...] = ("geometry", "alignment"),
        score_key: str = "score",
        aggregation_fn: Callable[[list[float]], float] | None = None,
        negate: bool = False,
        save_path: str | None = None,
        human_ranking: list[str] | None = None,
) -> dict:
    score_source = aggregation_fn.__name__ if aggregation_fn is not None else score_key

    per_criterion = {
        crit: {"human_outcome": [], "score_diff": []}
        for crit in criteria
    }

    for bid in scores:
        battle_gt = battles_data[bid]
        model_0, model_1 = battle_gt["models"][0], battle_gt["models"][1]
        err = 0
        if aggregation_fn is None:
            s0 = float(scores[bid][model_0][score_key])
            s1 = float(scores[bid][model_1][score_key])
        else:
            try:
                if "per_pair_scores" in scores[bid][model_0]:
                    s0 = float(aggregation_fn(scores[bid][model_0]["per_pair_scores"]))
                    s1 = float(aggregation_fn(scores[bid][model_1]["per_pair_scores"]))
                elif "per_view_mean" in scores[bid][model_0]:
                    s0 = float(aggregation_fn(scores[bid][model_0]["per_view_mean"]))
                    s1 = float(aggregation_fn(scores[bid][model_1]["per_view_mean"]))
                elif "per_view_scores" in scores[bid][model_0]:
                    s0 = float(aggregation_fn(scores[bid][model_0]["per_view_scores"]))
                    s1 = float(aggregation_fn(scores[bid][model_1]["per_view_scores"]))
                elif "per_dim_scores" in scores[bid][model_0]:
                    s0 = float(aggregation_fn(scores[bid][model_0]["per_dim_scores"]))
                    s1 = float(aggregation_fn(scores[bid][model_1]["per_dim_scores"]))
                else:
                    raise ValueError(f"No per-pair or per-view scores found for {model_0} or {model_1} in {bid}")
            except KeyError as k:
                err += 1
                continue
        assert err < 10, "many errors"
        effective_s0 = -s0 if negate else s0
        effective_s1 = -s1 if negate else s1
        score_diff = effective_s0 - effective_s1

        for crit in criteria:
            label_0 = battle_gt[crit]["label_0"]
            label_1 = battle_gt[crit]["label_1"]
            per_criterion[crit]["human_outcome"].append(_battle_outcome(label_0, label_1))
            per_criterion[crit]["score_diff"].append(score_diff)

    results = {
        "score_key": score_key,
        "score_source": score_source,
        "negate": negate,
        "metrics": {},
    }

    for crit in criteria:
        y_true = np.asarray(per_criterion[crit]["human_outcome"], dtype=int)
        score_diff = np.asarray(per_criterion[crit]["score_diff"], dtype=float)
        n_samples = len(y_true)

        auc_mask = y_true != 1  # discard ties
        y_auc = (y_true[auc_mask] == 2).astype(int)
        score_auc = score_diff[auc_mask]

        if len(np.unique(y_auc)) < 2:
            raise ValueError(f"Cannot compute AUC-ROC for {crit}: only one human class remains after dropping ties.")

        auc_roc = roc_auc_score(y_auc, score_auc)
        absolute_accuracy = np.mean((score_auc > 0).astype(int) == y_auc)

        results["metrics"][crit] = {
            "auc_roc": round(float(auc_roc), 3),
            "absolute_accuracy": round(float(absolute_accuracy), 3),
            "n_samples": int(n_samples),
            "n_auc_samples": int(len(y_auc)),
            "n_ties_dropped": int(n_samples - len(y_auc)),
        }

    if save_path:
        with open(save_path, "w") as f:
            # json.dump(results, f, indent=2, default=make_serializable)
            json.dump(results, f, indent=2)

    return results


BATTLES_JSON_PATH_TEST = f"../data/assets/3dgenbench_processed_12_views/test.json"

with open(BATTLES_JSON_PATH_TEST, "r") as f:
    battles_data_test = json.load(f)

RUN_SPECS_TEST = [

    ("MEt3R", "met3r_test", True, "met3r_full.json", "score"),
    ("3DGenScore-Geom", "3dgenscore_test", False, "3dgenscore_full.json", lambda x: x[0]),  # geometry
    ("3DGenScore-Align", "3dgenscore_test", False, "3dgenscore_full.json", lambda x: x[4]),  # alignment

    ("Hi3DEval-Geom", "hi3deval_test", False, "hi3deval_full.json", lambda x: x["geometry"]),  # geometry
    ("Hi3DEval-Align", "hi3deval_test", False, "hi3deval_full.json", lambda x: x["alignment"]),

    ("M3aSURE-Zero", "projector_test", False, "projector_full.json", "long_range_score"),
    ("M3aSURE-One", "latent_alignment_test", False, "latent_alignment_full.json", "mean_score"),
]


def _parse_continuous_run_spec(run_spec: tuple) -> tuple[
    str, str, bool, str, str, Callable[[list[float]], float] | None]:
    if len(run_spec) == 3:
        variant_name, run_id, negate = run_spec
        return variant_name, run_id, negate, _infer_full_json_name(run_id), "score", None
    if len(run_spec) == 4:
        variant_name, run_id, negate, full_json_name = run_spec
        return variant_name, run_id, negate, full_json_name, "score", None
    if len(run_spec) == 5:
        variant_name, run_id, negate, full_json_name, score_source = run_spec
        if callable(score_source):
            return variant_name, run_id, negate, full_json_name, "score", score_source
        return variant_name, run_id, negate, full_json_name, score_source, None
    raise ValueError(
        "Run specs must be (variant, run_id, negate), (variant, run_id, negate, full_json), or (variant, run_id, negate, full_json, score_key_or_aggregation_fn).")


def compare_continuous_3dgen_metrics(
        run_specs: list[tuple],
        battles_data: dict,
        results_root: str = "../results/3d_gen_bench",
        criteria: tuple[str, ...] = ("geometry", "alignment"),
        save_csv: str | None = None,
) -> pd.DataFrame:
    """Aggregate continuous 3D-gen metrics across runs into a pandas table."""

    prefix_map = {"geometry": "geom", "alignment": "align"}

    all_results = {}
    variant_order = []

    for run_spec in run_specs:
        variant_name, run_id, negate, full_json_name, score_key, aggregation_fn = _parse_continuous_run_spec(run_spec)
        full_path = os.path.join(results_root, run_id, full_json_name)

        with open(full_path, "r") as f:
            scores = json.load(f)

        results = compute_3dgen_metrics_continuous(
            scores=scores,
            battles_data=battles_data,
            criteria=criteria,
            score_key=score_key,
            aggregation_fn=aggregation_fn,
            negate=negate,
        )

        all_results[variant_name] = results
        variant_order.append(variant_name)

    def merge_scores(base_name, geom_name, align_name):
        if geom_name in all_results and align_name in all_results:
            merged = {
                "score_key": "merged",
                "score_source": "merged",
                "negate": False,
                "metrics": {
                    "geometry": all_results[geom_name]["metrics"]["geometry"],
                    "alignment": all_results[align_name]["metrics"]["alignment"],
                }
            }
            all_results[base_name] = merged

            idx = variant_order.index(geom_name)
            variant_order[idx] = base_name
            variant_order.remove(align_name)

            del all_results[geom_name]
            del all_results[align_name]

    merge_scores("3DGenScore", "3DGenScore-Geom", "3DGenScore-Align")
    merge_scores("Hi3DEval", "Hi3DEval-Geom", "Hi3DEval-Align")

    rows = []
    for variant_name in variant_order:
        results = all_results[variant_name]
        row = {"Metric": variant_name}
        for crit in criteria:
            prefix = prefix_map[crit]
            if crit in results["metrics"]:
                metrics = results["metrics"][crit]

                prefix_print = prefix.capitalize()
                row[f"{prefix_print} AUC $\\uparrow$"] = f"{metrics['auc_roc']:.3f}"
                row[f"{prefix_print} Acc. $\\uparrow$"] = f"{metrics['absolute_accuracy']:.3f}"
                row[f"{prefix_print} N"] = metrics["n_samples"]
                # row[f"{prefix_print} AUC N"] = metrics["n_auc_samples"]

        rows.append(row)

    df = pd.DataFrame(rows)

    if save_csv:
        save_dir = os.path.dirname(save_csv)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        df.to_csv(save_csv, index=False)

    # reorder
    df = df[
        ["Metric", "Geom AUC $\\uparrow$", "Align AUC $\\uparrow$", "Geom Acc. $\\uparrow$", "Align Acc. $\\uparrow$"]]
    return df


def format_latex_table_cvpr_custom(df: pd.DataFrame) -> str:
    # Mappings for metric names (adding citations and bolding our metrics)
    name_mapping = {
        "MEt3R": r"MEt3R \cite{asim_met3r_2026}",
        "3DGenScore": r"3DGenScore \cite{zhang_3dgen-bench_2025}",
        "Hi3DEval": r"Hi3DEval \cite{zhang_hi3deval_2025}",
        "M3aSURE-Zero": r"\textbf{M3aSURE-Zero}",
        "M3aSURE-One": r"\textbf{M3aSURE-One}",
    }

    our_metrics = ["M3aSURE-Zero", "M3aSURE-One"]

    cols_to_max = [
        "Geom AUC $\\uparrow$", "Align AUC $\\uparrow$",
        "Geom Acc. $\\uparrow$", "Align Acc. $\\uparrow$"
    ]

    # Find max values for each column
    max_vals = {}
    for col in cols_to_max:
        max_vals[col] = df[col].astype(float).max()

    lines = []
    lines.append(r"\begin{tabular}{l cc cc}")
    lines.append(r"    \toprule")
    lines.append(r"    & \multicolumn{2}{c}{\textbf{AUC $\uparrow$}} & \multicolumn{2}{c}{\textbf{Acc. $\uparrow$}} \\")
    lines.append(r"    \cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    lines.append(r"    \textbf{Metric} & Geom. & Align. & Geom. & Align. \\")
    lines.append(r"    \midrule")

    has_rendered_ours = False

    for _, row in df.iterrows():
        metric = row["Metric"]

        # Add a midrule before our metrics group
        if metric in our_metrics and not has_rendered_ours:
            lines.append(r"    \midrule")
            has_rendered_ours = True

        metric_name = name_mapping.get(metric, metric)
        formatted_row = [metric_name]

        for col in cols_to_max:
            val_str = row[col]
            val_float = float(val_str)

            if abs(val_float - max_vals[col]) < 1e-6:
                formatted_row.append(f"\\textbf{{{val_str}}}")
            else:
                formatted_row.append(val_str)

        lines.append("    " + " & ".join(formatted_row) + r" \\")

    lines.append(r"    \bottomrule")
    lines.append(r"\end{tabular}")

    return "\n".join(lines)


if __name__ == "__main__":
    prt = False
    df = compare_continuous_3dgen_metrics(
        RUN_SPECS_TEST,
        battles_data_test,
    )
    if prt:
        print(tabulate(df))
    else:
        print(format_latex_table_cvpr_custom(df))
