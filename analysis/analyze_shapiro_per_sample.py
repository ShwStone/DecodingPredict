"""Run per-sample Shapiro-Wilk tests and save a Q-Q plot for the max-std sample."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

try:
    from scipy import stats
except ImportError as exc:  # pragma: no cover - runtime guard
    raise RuntimeError(
        "scipy is required for Shapiro-Wilk analysis. Install scipy first."
    ) from exc


DEFAULT_INPUT_PATH = Path(
    "analysis_outputs/Qwen_Qwen3-32B+no-thinking/sharegpt_std/per_sample_stats_20260315_013136.jsonl"
)
DEFAULT_ALPHA = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Shapiro-Wilk tests per sample and save Q-Q plot"
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
        default="shapiro_analysis",
        help="Output filename prefix",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Significance level for normality decision",
    )
    return parser.parse_args()


def load_records(input_path: Path) -> list[dict[str, Any]]:
    if not input_path.exists():
        raise RuntimeError(f"Input file does not exist: {input_path}")

    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        raise RuntimeError(f"No valid records found in: {input_path}")
    return records


def run_shapiro(lengths: list[float], alpha: float) -> dict[str, Any]:
    if len(lengths) < 3:
        return {
            "shapiro_valid": False,
            "shapiro_reason": "need at least 3 observations",
            "shapiro_w": None,
            "shapiro_p": None,
            "is_normal_at_alpha": None,
            "shapiro_warning": None,
        }

    if min(lengths) == max(lengths):
        return {
            "shapiro_valid": False,
            "shapiro_reason": "all observations are identical",
            "shapiro_w": None,
            "shapiro_p": None,
            "is_normal_at_alpha": None,
            "shapiro_warning": None,
        }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        statistic, p_value = stats.shapiro(lengths)

    warning_message = str(caught[0].message) if caught else None
    return {
        "shapiro_valid": True,
        "shapiro_reason": None,
        "shapiro_w": float(statistic),
        "shapiro_p": float(p_value),
        "is_normal_at_alpha": bool(p_value >= alpha),
        "shapiro_warning": warning_message,
    }


def build_results(records: list[dict[str, Any]], alpha: float) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for record in records:
        lengths = [float(v) for v in record.get("lengths", [])]
        test_result = run_shapiro(lengths, alpha)
        results.append(
            {
                "sample_index": int(record["sample_index"]),
                "conversation_id": record.get("conversation_id"),
                "num_successful_runs": int(record.get("num_successful_runs", len(lengths))),
                "mean": float(record.get("mean", 0.0)),
                "std": float(record.get("std", 0.0)),
                **test_result,
            }
        )
    return results


def save_results_jsonl(path: Path, results: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for item in results:
            fp.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def save_summary_json(path: Path, results: list[dict[str, Any]], alpha: float) -> None:
    valid_results = [item for item in results if item["shapiro_valid"]]
    normal_count = sum(1 for item in valid_results if item["is_normal_at_alpha"])
    summary = {
        "num_samples": len(results),
        "num_valid_shapiro_samples": len(valid_results),
        "num_invalid_shapiro_samples": len(results) - len(valid_results),
        "alpha": alpha,
        "num_not_reject_normality": normal_count,
        "num_reject_normality": len(valid_results) - normal_count,
    }
    with path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)


def save_qq_plot(path: Path, record: dict[str, Any]) -> None:
    lengths = [float(v) for v in record.get("lengths", [])]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    stats.probplot(lengths, dist="norm", plot=ax)
    ax.set_title(
        f"Q-Q Plot for sample_{int(record['sample_index']):06d} (std={float(record['std']):.4f})"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    records = load_records(args.input_path)
    results = build_results(records, args.alpha)

    output_dir = args.output_dir or args.input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = args.input_path.stem
    results_path = output_dir / f"{args.prefix}_{base_name}.jsonl"
    summary_path = output_dir / f"{args.prefix}_{base_name}.json"

    import random
    max_std_record = max(records, key=lambda item: float(random.random()))
    qq_plot_path = output_dir / f"{args.prefix}_{base_name}_max_std_qq.png"

    save_results_jsonl(results_path, results)
    save_summary_json(summary_path, results, args.alpha)
    save_qq_plot(qq_plot_path, max_std_record)

    print(f"Input file: {args.input_path}")
    print(f"Per-sample Shapiro results: {results_path}")
    print(f"Summary json: {summary_path}")
    print(
        "Max-std sample for Q-Q plot: "
        f"sample_{int(max_std_record['sample_index']):06d} "
        f"(std={float(max_std_record['std']):.6f})"
    )
    print(f"Q-Q plot: {qq_plot_path}")


if __name__ == "__main__":
    main()