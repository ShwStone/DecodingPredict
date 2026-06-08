import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class LengthRegressorMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, chunk_norm_size: int = 5120):
        super().__init__()
        self.chunk_norm_size = int(chunk_norm_size)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.reg_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        out = self.reg_head(features)
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LengthRegressorMLP GPU latency")
    parser.add_argument(
        "--input-dim",
        type=int,
        required=True,
        help="Input feature dimension for the random benchmark tensor",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension for LengthRegressorMLP",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=20,
        help="Warmup iterations before latency measurement",
    )
    parser.add_argument(
        "--measure-iters",
        type=int,
        default=50,
        help="Measurement iterations used for averaging",
    )
    return parser.parse_args()


def benchmark_batch_latency(
    model: nn.Module,
    input_dim: int,
    batch_size: int,
    warmup_iters: int,
    measure_iters: int,
    device: torch.device,
) -> tuple[float, float]:
    x = torch.randn(batch_size, input_dim, device=device, dtype=torch.float32)

    with torch.inference_mode():
        for _ in range(warmup_iters):
            _ = model(x)
        torch.cuda.synchronize(device)

        total_seconds = 0.0
        for _ in range(measure_iters):
            start = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize(device)
            end = time.perf_counter()
            total_seconds += end - start

    avg_step_ms = (total_seconds / measure_iters) * 1000.0
    avg_per_sample_ms = avg_step_ms / float(batch_size)
    return avg_step_ms, avg_per_sample_ms


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark must run on GPU.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LengthRegressorMLP(input_dim=args.input_dim, hidden_dim=args.hidden_dim).to(device)
    model.eval()

    batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    print(f"device: {torch.cuda.get_device_name(device)}")
    print(
        f"input_dim={args.input_dim}, hidden_dim={args.hidden_dim}, "
        f"warmup_iters={args.warmup_iters}, measure_iters={args.measure_iters}"
    )
    print("batch_size\tavg_step_latency_ms\tavg_per_sample_latency_ms")

    for batch_size in batch_sizes:
        avg_step_ms, avg_per_sample_ms = benchmark_batch_latency(
            model=model,
            input_dim=args.input_dim,
            batch_size=batch_size,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
            device=device,
        )
        print(f"{batch_size}\t\t{avg_step_ms:.4f}\t\t\t{avg_per_sample_ms:.4f}")


if __name__ == "__main__":
    main()


