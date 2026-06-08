"""Analyze correlation between mean and std from per-sample stats jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT_PATH = Path(
    "analysis_outputs/Qwen_Qwen3-32B+no-thinking/sharegpt_std/per_sample_stats_20260315_013136.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute mean/std correlation and draw a curve plot"
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to per_sample_stats_*.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same directory as input)",
    )
    parser.add_argument(
        "--prefix",
        default="mean_std_corr",
        help="Output filename prefix",
    )
    return parser.parse_args()


def load_mean_std(jsonl_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not jsonl_path.exists():
        raise RuntimeError(f"Input file does not exist: {jsonl_path}")

    means: list[float] = []
    stds: list[float] = []
    sample_ids: list[int] = []
    with jsonl_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample_id = record.get("sample_index")
            mean_value = record.get("mean")
            std_value = record.get("std")
            if sample_id is None or mean_value is None or std_value is None:
                continue
            sample_ids.append(int(sample_id))
            means.append(float(mean_value))
            stds.append(float(std_value))

    if not means:
        raise RuntimeError("No valid mean/std records found in input file")

    return (
        np.array(sample_ids, dtype=np.int64),
        np.array(means, dtype=np.float64),
        np.array(stds, dtype=np.float64),
    )


def compute_stats(sample_ids: np.ndarray, means: np.ndarray, stds: np.ndarray) -> dict:
    if len(means) < 2:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(means, stds)[0, 1])

    linear_coeff = np.polyfit(means, stds, 1) if len(means) >= 2 else np.array([np.nan, np.nan])
    slope = float(linear_coeff[0])
    intercept = float(linear_coeff[1])

    q_lower = 0.025
    q_upper = 0.975
    lower_slope, lower_intercept = fit_quantile_line(means, stds, q_lower)
    upper_slope, upper_intercept = fit_quantile_line(means, stds, q_upper)

    y_lower = lower_slope * means + lower_intercept
    y_upper = upper_slope * means + upper_intercept
    lo = np.minimum(y_lower, y_upper)
    hi = np.maximum(y_lower, y_upper)
    covered = (stds >= lo) & (stds <= hi)
    envelope_coverage = float(np.mean(covered))
    uncovered_sample_ids = sample_ids[~covered].astype(np.int64).tolist()
    uncovered_upper_ids = sample_ids[stds > hi].astype(np.int64).tolist()
    uncovered_lower_ids = sample_ids[stds < lo].astype(np.int64).tolist()

    return {
        "num_samples": int(len(means)),
        "mean_of_mean": float(np.mean(means)),
        "mean_of_std": float(np.mean(stds)),
        "min_mean": float(np.min(means)),
        "max_mean": float(np.max(means)),
        "min_std": float(np.min(stds)),
        "max_std": float(np.max(stds)),
        "pearson_corr_mean_std": pearson,
        "linear_fit_slope": slope,
        "linear_fit_intercept": intercept,
        "envelope_method": "linear_quantile_regression",
        "envelope_lower_quantile": q_lower,
        "envelope_upper_quantile": q_upper,
        "envelope_lower_slope": float(lower_slope),
        "envelope_lower_intercept": float(lower_intercept),
        "envelope_upper_slope": float(upper_slope),
        "envelope_upper_intercept": float(upper_intercept),
        "envelope_target_coverage": 0.95,
        "envelope_actual_coverage": envelope_coverage,
        "num_uncovered_samples": int(len(uncovered_sample_ids)),
        "uncovered_sample_ids": uncovered_sample_ids,
        "num_uncovered_upper_samples": int(len(uncovered_upper_ids)),
        "uncovered_upper_sample_ids": uncovered_upper_ids,
        "num_uncovered_lower_samples": int(len(uncovered_lower_ids)),
        "uncovered_lower_sample_ids": uncovered_lower_ids,
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
        # Pinball loss subgradient for d/d(y_hat)
        grad_base = np.where(residual > 0.0, -q, 1.0 - q)
        grad_slope = float(np.mean(grad_base * x_n))
        grad_intercept = float(np.mean(grad_base))
        slope_n -= lr * grad_slope
        intercept_n -= lr * grad_intercept

    slope = slope_n * (y_scale / x_scale)
    intercept = y_mean + y_scale * intercept_n - slope * x_mean
    return float(slope), float(intercept)


def plot_curve(means: np.ndarray, stds: np.ndarray, stats: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.scatter(means, stds, s=14, alpha=0.65, color="#ff7f0e", edgecolors="none")

    plotted_envelope = False
    if np.isfinite(stats["envelope_lower_slope"]) and np.isfinite(stats["envelope_upper_slope"]):
        x_line = np.linspace(float(np.min(means)), float(np.max(means)), 200)
        y_lower = stats["envelope_lower_slope"] * x_line + stats["envelope_lower_intercept"]
        y_upper = stats["envelope_upper_slope"] * x_line + stats["envelope_upper_intercept"]
        lo = np.minimum(y_lower, y_upper)
        hi = np.maximum(y_lower, y_upper)
        ax.fill_between(
            x_line,
            lo,
            hi,
            color="#9ecae1",
            alpha=0.22,
            label="95% envelope band",
        )
        ax.plot(
            x_line,
            y_lower,
            color="#1f77b4",
            linewidth=2.0,
            linestyle="--",
            alpha=0.85,
            label=(
                "lower quantile: y="
                f"{stats['envelope_lower_slope']:.4f}x+{stats['envelope_lower_intercept']:.4f}"
            ),
        )
        ax.plot(
            x_line,
            y_upper,
            color="#2ca02c",
            linewidth=2.0,
            linestyle="--",
            alpha=0.85,
            label=(
                "upper quantile: y="
                f"{stats['envelope_upper_slope']:.4f}x+{stats['envelope_upper_intercept']:.4f}"
            ),
        )
        plotted_envelope = True

    if np.isfinite(stats["linear_fit_slope"]) and np.isfinite(stats["linear_fit_intercept"]):
        x_reg = np.linspace(float(np.min(means)), float(np.max(means)), 200)
        y_reg = stats["linear_fit_slope"] * x_reg + stats["linear_fit_intercept"]
        ax.plot(
            x_reg,
            y_reg,
            color="#d62728",
            linewidth=2.0,
            alpha=0.9,
            label=(
                "linear fit: y="
                f"{stats['linear_fit_slope']:.4f}x+{stats['linear_fit_intercept']:.4f}"
            ),
        )

    ax.set_title(
        "Mean-Std Scatter with 95% Envelope "
        f"(r={stats['pearson_corr_mean_std']:.4f}, coverage={stats['envelope_actual_coverage']:.3f})"
    )
    ax.set_xlabel("mean")
    ax.set_ylabel("std")
    ax.grid(alpha=0.3)
    if plotted_envelope or (
        np.isfinite(stats["linear_fit_slope"]) and np.isfinite(stats["linear_fit_intercept"])
    ):
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    sample_ids, means, stds = load_mean_std(args.input_path)
    stats = compute_stats(sample_ids, means, stds)

    output_dir = args.output_dir or args.input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = args.input_path.stem
    stats_path = output_dir / f"{args.prefix}_{base_name}.json"
    figure_path = output_dir / f"{args.prefix}_{base_name}.png"

    plot_curve(means, stds, stats, figure_path)
    with stats_path.open("w", encoding="utf-8") as fp:
        json.dump(stats, fp, ensure_ascii=False, indent=2)

    print(f"Input file: {args.input_path}")
    print(f"Num samples: {stats['num_samples']}")
    print(f"Pearson corr(mean, std): {stats['pearson_corr_mean_std']:.6f}")
    print(f"Stats saved to: {stats_path}")
    print(f"Figure saved to: {figure_path}")


if __name__ == "__main__":
    main()