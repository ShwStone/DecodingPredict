"""Analyze relationships between model errors and theoretical per-sample statistics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TEST_MAE_PATH = Path(
    "prediction/log_regression/best_model_run_10_test_sample_mae.jsonl"
)
DEFAULT_STD_PATH = Path(
    "analysis_outputs/Qwen_Qwen3-32B+no-thinking/sharegpt_std/per_sample_stats_20260315_013136.jsonl"
)

DEFAULT_ENVELOPE_LOWER_QUANTILE = 0.025
DEFAULT_ENVELOPE_UPPER_QUANTILE = 0.975


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze correlations between prediction errors and theoretical statistics"
    )
    parser.add_argument(
        "--test-mae-path",
        type=Path,
        default=DEFAULT_TEST_MAE_PATH,
        help="Path to best_model_run_*_test_sample_mae.jsonl",
    )
    parser.add_argument(
        "--std-path",
        type=Path,
        default=DEFAULT_STD_PATH,
        help="Path to per_sample_stats_*.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same directory as test-mae-path)",
    )
    parser.add_argument(
        "--prefix",
        default="test_mae_vs_std",
        help="Output filename prefix",
    )
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Input file does not exist: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        raise RuntimeError(f"No valid records found in: {path}")
    return records


def normalize_sample_id(value: str) -> int:
    if value.startswith("sample_"):
        return int(value.split("_")[-1])
    return int(value)


def build_std_lookup(std_records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for record in std_records:
        sample_index = record.get("sample_index")
        std_value = record.get("std")
        if sample_index is None or std_value is None:
            continue
        lookup[int(sample_index)] = record
    return lookup


def merge_records(
    test_records: list[dict[str, Any]], std_lookup: dict[int, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    merged: list[dict[str, Any]] = []
    missing_ids: list[str] = []

    for record in test_records:
        sample_id = record.get("sample_id")
        mae = record.get("mae")
        pred = record.get("pred")
        target = record.get("target")
        if sample_id is None or mae is None:
            continue

        sample_index = normalize_sample_id(str(sample_id))
        std_record = std_lookup.get(sample_index)
        if std_record is None:
            missing_ids.append(str(sample_id))
            continue

        merged.append(
            {
                "sample_id": str(sample_id),
                "sample_index": sample_index,
                "test_mae": float(mae),
                "pred": safe_float(pred),
                "target": safe_float(target),
                "std": float(std_record["std"]),
                "mean": float(std_record.get("mean", 0.0)),
                "pred_minus_theoretical_mean": (
                    float(pred) - float(std_record.get("mean", 0.0))
                    if pred is not None
                    else None
                ),
                "abs_pred_minus_theoretical_mean": (
                    abs(float(pred) - float(std_record.get("mean", 0.0)))
                    if pred is not None
                    else None
                ),
                "acc100_upper_bound": float(std_record.get("acc100_upper_bound", 0.0)),
                "acc200_upper_bound": float(std_record.get("acc200_upper_bound", 0.0)),
                "acc_x_90": float(std_record.get("acc_x_90", 0.0)),
            }
        )

    if not merged:
        raise RuntimeError("No overlapping samples found between test MAE and std records.")
    return merged, missing_ids


def get_xy_from_merged(
    merged: list[dict[str, Any]], x_key: str, y_key: str
) -> tuple[np.ndarray, np.ndarray]:
    filtered = [
        item
        for item in merged
        if item.get(x_key) is not None and item.get(y_key) is not None
    ]
    x = np.array([float(item[x_key]) for item in filtered], dtype=np.float64)
    y = np.array([float(item[y_key]) for item in filtered], dtype=np.float64)
    return x, y


def compute_relation_stats(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_key: str,
    y_key: str,
    theory_line: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pearson = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 else float("nan")
    coeff = np.polyfit(x, y, 1) if len(x) >= 2 else np.array([np.nan, np.nan])
    slope = float(coeff[0])
    intercept = float(coeff[1])

    q_lower = DEFAULT_ENVELOPE_LOWER_QUANTILE
    q_upper = DEFAULT_ENVELOPE_UPPER_QUANTILE
    lower_slope, lower_intercept = fit_quantile_line(x, y, q_lower)
    upper_slope, upper_intercept = fit_quantile_line(x, y, q_upper)

    y_lower = lower_slope * x + lower_intercept
    y_upper = upper_slope * x + upper_intercept
    lo = np.minimum(y_lower, y_upper)
    hi = np.maximum(y_lower, y_upper)
    covered = (y >= lo) & (y <= hi)
    envelope_coverage = float(np.mean(covered)) if len(covered) > 0 else float("nan")

    stats: dict[str, Any] = {
        "x_key": x_key,
        "y_key": y_key,
        "num_samples": int(len(x)),
        "pearson_corr": pearson,
        "linear_fit_slope": slope,
        "linear_fit_intercept": intercept,
        "envelope_method": "linear_quantile_regression",
        "envelope_lower_quantile": q_lower,
        "envelope_upper_quantile": q_upper,
        "envelope_lower_slope": float(lower_slope),
        "envelope_lower_intercept": float(lower_intercept),
        "envelope_upper_slope": float(upper_slope),
        "envelope_upper_intercept": float(upper_intercept),
        "envelope_target_coverage": q_upper - q_lower,
        "envelope_actual_coverage": envelope_coverage,
        "mean_x": float(np.mean(x)),
        "mean_y": float(np.mean(y)),
        "min_x": float(np.min(x)),
        "max_x": float(np.max(x)),
        "min_y": float(np.min(y)),
        "max_y": float(np.max(y)),
    }
    if theory_line is not None:
        stats["theory_line"] = theory_line
    return stats


def compute_stats(merged: list[dict[str, Any]]) -> dict[str, Any]:
    pred_minus_theoretical_mean = np.array(
        [
            item["pred_minus_theoretical_mean"]
            for item in merged
            if item["pred_minus_theoretical_mean"] is not None
        ],
        dtype=np.float64,
    )
    abs_pred_minus_theoretical_mean = np.array(
        [
            item["abs_pred_minus_theoretical_mean"]
            for item in merged
            if item["abs_pred_minus_theoretical_mean"] is not None
        ],
        dtype=np.float64,
    )

    relation_test_mae_vs_std = compute_relation_stats(
        *get_xy_from_merged(merged, "std", "test_mae"),
        x_key="std",
        y_key="test_mae",
        theory_line={
            "form": "mae = sqrt(2/pi) * std",
            "slope": math.sqrt(2.0 / math.pi),
            "intercept": 0.0,
        },
    )
    relation_abs_pred_mean_vs_std = compute_relation_stats(
        *get_xy_from_merged(merged, "std", "abs_pred_minus_theoretical_mean"),
        x_key="std",
        y_key="abs_pred_minus_theoretical_mean",
    )
    relation_abs_pred_mean_vs_mean = compute_relation_stats(
        *get_xy_from_merged(merged, "mean", "abs_pred_minus_theoretical_mean"),
        x_key="mean",
        y_key="abs_pred_minus_theoretical_mean",
    )

    return {
        "num_samples": len(merged),
        "pearson_corr_std_test_mae": relation_test_mae_vs_std["pearson_corr"],
        "linear_fit_slope": relation_test_mae_vs_std["linear_fit_slope"],
        "linear_fit_intercept": relation_test_mae_vs_std["linear_fit_intercept"],
        "theory_line_form": relation_test_mae_vs_std["theory_line"]["form"],
        "theory_line_slope": relation_test_mae_vs_std["theory_line"]["slope"],
        "envelope_method": relation_test_mae_vs_std["envelope_method"],
        "envelope_lower_quantile": relation_test_mae_vs_std["envelope_lower_quantile"],
        "envelope_upper_quantile": relation_test_mae_vs_std["envelope_upper_quantile"],
        "envelope_lower_slope": relation_test_mae_vs_std["envelope_lower_slope"],
        "envelope_lower_intercept": relation_test_mae_vs_std["envelope_lower_intercept"],
        "envelope_upper_slope": relation_test_mae_vs_std["envelope_upper_slope"],
        "envelope_upper_intercept": relation_test_mae_vs_std["envelope_upper_intercept"],
        "envelope_target_coverage": relation_test_mae_vs_std["envelope_target_coverage"],
        "envelope_actual_coverage": relation_test_mae_vs_std["envelope_actual_coverage"],
        "mean_test_mae": relation_test_mae_vs_std["mean_y"],
        "mean_pred_minus_theoretical_mean": (
            float(np.mean(pred_minus_theoretical_mean))
            if len(pred_minus_theoretical_mean) > 0
            else None
        ),
        "mean_abs_pred_minus_theoretical_mean": (
            float(np.mean(abs_pred_minus_theoretical_mean))
            if len(abs_pred_minus_theoretical_mean) > 0
            else None
        ),
        "mean_std": relation_test_mae_vs_std["mean_x"],
        "min_test_mae": relation_test_mae_vs_std["min_y"],
        "max_test_mae": relation_test_mae_vs_std["max_y"],
        "min_std": relation_test_mae_vs_std["min_x"],
        "max_std": relation_test_mae_vs_std["max_x"],
        "analyses": {
            "test_mae_vs_std": relation_test_mae_vs_std,
            "abs_pred_minus_theoretical_mean_vs_std": relation_abs_pred_mean_vs_std,
            "abs_pred_minus_theoretical_mean_vs_mean": relation_abs_pred_mean_vs_mean,
        },
    }


def fit_quantile_line(
    x: np.ndarray,
    y: np.ndarray,
    q: float,
    steps: int = 4000,
    lr: float = 0.05,
) -> tuple[float, float]:
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) == 1:
        return 0.0, float(y[0])

    x_mean = float(np.mean(x))
    x_std = float(np.std(x))
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))

    x_scale = x_std if x_std > 1e-12 else 1.0
    y_scale = y_std if y_std > 1e-12 else 1.0

    x_n = (x - x_mean) / x_scale
    y_n = (y - y_mean) / y_scale

    ols = np.polyfit(x_n, y_n, 1)
    slope_n = float(ols[0])
    intercept_n = float(ols[1])

    for _ in range(steps):
        residual = y_n - (slope_n * x_n + intercept_n)
        grad_base = np.where(residual > 0.0, -q, 1.0 - q)
        grad_slope = float(np.mean(grad_base * x_n))
        grad_intercept = float(np.mean(grad_base))
        slope_n -= lr * grad_slope
        intercept_n -= lr * grad_intercept

    slope = slope_n * (y_scale / x_scale)
    intercept = y_mean + y_scale * intercept_n - slope * x_mean
    return float(slope), float(intercept)


def save_merged_jsonl(path: Path, merged: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for item in merged:
            fp.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def save_relation_plot(
    path: Path,
    merged: list[dict[str, Any]],
    relation_stats: dict[str, Any],
    *,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    x, y = get_xy_from_merged(merged, x_key, y_key)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Highlight the MAE<=100 region to visually indicate acc@100.
    is_mae_y_axis = ("mae" in y_key.lower()) or ("mae" in y_label.lower())
    if is_mae_y_axis:
        ax.axhspan(
            0.0,
            100.0,
            color="#ffd54f",
            alpha=0.25,
            zorder=0,
            label="acc@100 (MAE<=100)",
        )

    ax.scatter(x, y, s=18, alpha=0.7, color="#ff7f0e", edgecolors="none")

    x_line = np.linspace(float(np.min(x)), float(np.max(x)), 200)

    theory_line = relation_stats.get("theory_line")
    if theory_line is not None:
        y_theory = theory_line["slope"] * x_line + theory_line.get("intercept", 0.0)
        ax.plot(
            x_line,
            y_theory,
            color="#2ca02c",
            linewidth=2.0,
            linestyle="--",
            label=(
                "theory: y="
                f"{theory_line['slope']:.4f}x+{theory_line.get('intercept', 0.0):.4f}"
            ),
        )

    if np.isfinite(relation_stats["envelope_lower_slope"]) and np.isfinite(
        relation_stats["envelope_upper_slope"]
    ):
        envelope_lower = (
            relation_stats["envelope_lower_slope"] * x_line
            + relation_stats["envelope_lower_intercept"]
        )
        envelope_upper = (
            relation_stats["envelope_upper_slope"] * x_line
            + relation_stats["envelope_upper_intercept"]
        )
        lo = np.minimum(envelope_lower, envelope_upper)
        hi = np.maximum(envelope_lower, envelope_upper)
        ax.fill_between(
            x_line,
            lo,
            hi,
            color="#9ecae1",
            alpha=0.22,
            label="quantile envelope 95% band",
        )
        ax.plot(
            x_line,
            envelope_lower,
            color="#1f77b4",
            linewidth=2.0,
            linestyle="--",
            alpha=0.85,
            label=(
                "lower quantile: y="
                f"{relation_stats['envelope_lower_slope']:.4f}x+{relation_stats['envelope_lower_intercept']:.4f}"
            ),
        )
        ax.plot(
            x_line,
            envelope_upper,
            color="#9467bd",
            linewidth=2.0,
            linestyle="--",
            alpha=0.85,
            label=(
                "upper quantile: y="
                f"{relation_stats['envelope_upper_slope']:.4f}x+{relation_stats['envelope_upper_intercept']:.4f}"
            ),
        )

    if np.isfinite(relation_stats["linear_fit_slope"]) and np.isfinite(
        relation_stats["linear_fit_intercept"]
    ):
        y_line = relation_stats["linear_fit_slope"] * x_line + relation_stats["linear_fit_intercept"]
        ax.plot(
            x_line,
            y_line,
            color="#1f77b4",
            linewidth=2.0,
            label=(
                "linear fit: y="
                f"{relation_stats['linear_fit_slope']:.4f}x+{relation_stats['linear_fit_intercept']:.4f}"
            ),
        )
    ax.legend(loc="best")

    ax.set_title(
        f"{title} (r={relation_stats['pearson_corr']:.4f}, env_cov={relation_stats['envelope_actual_coverage']:.3f})"
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    test_records = load_jsonl(args.test_mae_path)
    std_records = load_jsonl(args.std_path)
    std_lookup = build_std_lookup(std_records)
    merged, missing_ids = merge_records(test_records, std_lookup)
    stats = compute_stats(merged)

    mean_of_mean = float(np.mean([item["mean"] for item in merged])) if merged else float("nan")
    mean_of_std = float(np.mean([item["std"] for item in merged])) if merged else float("nan")
    count_std_lt_200 = int(sum(1 for item in merged if item["std"] < 200.0))
    mean_of_acc100 = (
        float(np.mean([item["acc100_upper_bound"] for item in merged])) if merged else float("nan")
    )
    test_mae_lt_100 = [item for item in merged if item["test_mae"] < 100.0]
    mean_std_test_mae_lt_100 = (
        float(np.mean([item["std"] for item in test_mae_lt_100]))
        if test_mae_lt_100
        else float("nan")
    )

    stats["mean_of_mean"] = mean_of_mean
    stats["mean_of_std"] = mean_of_std
    stats["count_std_lt_200"] = count_std_lt_200
    stats["mean_of_acc100"] = mean_of_acc100
    stats["num_test_mae_lt_100"] = len(test_mae_lt_100)
    stats["mean_std_test_mae_lt_100"] = mean_std_test_mae_lt_100

    stats["test_mae_path"] = str(args.test_mae_path)
    stats["std_path"] = str(args.std_path)
    stats["num_missing_std_records"] = len(missing_ids)
    stats["missing_std_sample_ids"] = missing_ids

    output_dir = args.output_dir or args.test_mae_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_path = output_dir / f"{args.prefix}_merged.jsonl"
    stats_path = output_dir / f"{args.prefix}_stats.json"
    plot_path = output_dir / f"{args.prefix}_scatter.png"
    abs_vs_std_plot_path = output_dir / f"{args.prefix}_abs_pred_minus_theoretical_mean_vs_std_scatter.png"
    abs_vs_mean_plot_path = output_dir / f"{args.prefix}_abs_pred_minus_theoretical_mean_vs_mean_scatter.png"

    save_merged_jsonl(merged_path, merged)
    with stats_path.open("w", encoding="utf-8") as fp:
        json.dump(stats, fp, ensure_ascii=False, indent=2)
    save_relation_plot(
        plot_path,
        merged,
        stats["analyses"]["test_mae_vs_std"],
        x_key="std",
        y_key="test_mae",
        x_label="std",
        y_label="test MAE",
        title="Test MAE vs Std",
    )
    save_relation_plot(
        abs_vs_std_plot_path,
        merged,
        stats["analyses"]["abs_pred_minus_theoretical_mean_vs_std"],
        x_key="std",
        y_key="abs_pred_minus_theoretical_mean",
        x_label="std",
        y_label="abs(pred - theoretical mean)",
        title="abs(pred - theoretical mean) vs Std",
    )
    save_relation_plot(
        abs_vs_mean_plot_path,
        merged,
        stats["analyses"]["abs_pred_minus_theoretical_mean_vs_mean"],
        x_key="mean",
        y_key="abs_pred_minus_theoretical_mean",
        x_label="mean",
        y_label="abs(pred - theoretical mean)",
        title="abs(pred - theoretical mean) vs Mean",
    )

    print(f"Test MAE file: {args.test_mae_path}")
    print(f"Std file: {args.std_path}")
    print(f"Merged samples: {stats['num_samples']}")
    print(f"Pearson corr(std, test_mae): {stats['pearson_corr_std_test_mae']:.6f}")
    print(f"mean of mean: {mean_of_mean:.6f}")
    print(f"mean of std: {mean_of_std:.6f}")
    print(f"count (std < 200): {count_std_lt_200}")
    print(f"mean of acc100: {mean_of_acc100:.6f}")
    print(
        f"mean std for test_mae < 100 (n={len(test_mae_lt_100)}): "
        f"{mean_std_test_mae_lt_100:.6f}"
    )
    print(
        "Pearson corr(std, abs(pred-theoretical mean)): "
        f"{stats['analyses']['abs_pred_minus_theoretical_mean_vs_std']['pearson_corr']:.6f}"
    )
    print(
        "Pearson corr(mean, abs(pred-theoretical mean)): "
        f"{stats['analyses']['abs_pred_minus_theoretical_mean_vs_mean']['pearson_corr']:.6f}"
    )
    print(f"Merged jsonl: {merged_path}")
    print(f"Stats json: {stats_path}")
    print(f"Scatter plot: {plot_path}")
    print(f"Abs-vs-std scatter plot: {abs_vs_std_plot_path}")
    print(f"Abs-vs-mean scatter plot: {abs_vs_mean_plot_path}")


if __name__ == "__main__":
    main()
