from pathlib import Path
from typing import Iterable, Sequence, Tuple

import torch


def init_error_metric_tracker(x_quantile: float = 0.9) -> dict:
    return {
        "x_quantile": float(x_quantile),
        "total_abs_error": 0.0,
        "total_samples": 0,
        "within_100": 0,
        "within_200": 0,
        "within_500": 0,
        "abs_err_chunks": [],
    }


def update_error_metric_tracker(tracker: dict, abs_err: torch.Tensor) -> None:
    abs_err = abs_err.detach().reshape(-1)
    tracker["total_abs_error"] += abs_err.sum().item()
    tracker["total_samples"] += abs_err.numel()
    tracker["within_100"] += (abs_err < 100).sum().item()
    tracker["within_200"] += (abs_err < 200).sum().item()
    tracker["within_500"] += (abs_err < 500).sum().item()
    tracker["abs_err_chunks"].append(abs_err.to("cpu"))


def finalize_error_metric_tracker(tracker: dict) -> dict:
    total = tracker["total_samples"]
    if total == 0:
        return {
            "mae": 0.0,
            "acc100": 0.0,
            "acc200": 0.0,
            "acc500": 0.0,
            "accx": 0.0,
            "x_value": 0.0,
            "x_quantile": tracker["x_quantile"],
        }

    all_abs_err = torch.cat(tracker["abs_err_chunks"]) if tracker["abs_err_chunks"] else torch.zeros(0)
    x_value = float(torch.quantile(all_abs_err, tracker["x_quantile"]).item())
    accx = float((all_abs_err <= x_value).float().mean().item())

    return {
        "mae": tracker["total_abs_error"] / total,
        "acc100": tracker["within_100"] / total,
        "acc200": tracker["within_200"] / total,
        "acc500": tracker["within_500"] / total,
        "accx": accx,
        "x_value": x_value,
        "x_quantile": tracker["x_quantile"],
    }


def build_soft_labels(
    targets: torch.Tensor,
    num_bins: int,
    bin_size: int,
    max_value: int,
) -> torch.Tensor:
    targets = targets.squeeze(-1)
    soft = torch.zeros((targets.size(0), num_bins), device=targets.device, dtype=torch.float32)

    for i, y in enumerate(targets.tolist()):
        if y <= 0:
            soft[i, 0] = 1.0
            continue
        if y >= max_value:
            soft[i, -1] = 1.0
            continue

        idx = int(y // bin_size)
        idx = min(idx, num_bins - 1)
        upper = min((idx + 1) * bin_size, max_value)
        lower = idx * bin_size
        span = max(upper - lower, 1)
        frac = (y - lower) / span

        if idx == num_bins - 1:
            soft[i, idx] = 1.0
        else:
            soft[i, idx] = 1.0 - frac
            soft[i, idx + 1] = frac

    return soft


def build_bin_centers(num_bins: int, bin_size: int, max_value: int) -> torch.Tensor:
    centers = []
    for i in range(num_bins):
        lower = i * bin_size
        upper = min((i + 1) * bin_size, max_value)
        centers.append((lower + upper) / 2.0)
    return torch.tensor(centers, dtype=torch.float32)


def write_sample_mae_log(
    records: Iterable[Tuple[str, float, float, float]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("sample_id\tmae\tpred\ttarget\n")
        for sid, mae, pred, target in records:
            f.write(f"{sid}\t{mae:.6f}\t{pred:.6f}\t{target:.6f}\n")

