import argparse
import glob
import io
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from tqdm import tqdm

try:
    import webdataset as wds
except ImportError:
    wds = None

from model_regression import LengthRegressorMLP
from train_utils import (
    write_sample_mae_log,
    init_error_metric_tracker,
    update_error_metric_tracker,
    finalize_error_metric_tracker,
)

def get_args():
    parser = argparse.ArgumentParser(description="LLM Token Length Regressor Training")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/model_and_data/shwstone/data/Qwen_Qwen3-32B+no-thinking/sharegpt_outputs/train",
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        default="/home/model_and_data/shwstone/data/Qwen_Qwen3-32B+no-thinking/sharegpt_outputs/train2",
    )
    parser.add_argument("--data_backend", type=str, choices=["dir", "webdataset"], default="dir")
    parser.add_argument("--train_shards", type=str, default="")
    parser.add_argument("--val_shards", type=str, default="")
    parser.add_argument("--test_shards", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--wds_shuffle_buffer", type=int, default=2048)
    return parser.parse_args()


class TraceDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.samples = []

        tqdm.write(f"Scanning {data_dir} for samples...")
        for sample_path in sorted(self.data_dir.rglob("sample_*")):
            if not sample_path.is_dir():
                continue

            decode_meta_path = sample_path / "decode" / "decode_metadata.json"
            probe_path = sample_path / "prefill" / "prefill_concat_last_token_hidden.pt"

            if decode_meta_path.exists() and probe_path.exists():
                self.samples.append(
                    {
                        "sample_id": sample_path.name,
                        "meta_path": str(decode_meta_path),
                        "probe_path": str(probe_path),
                    }
                )

        tqdm.write(f"Found {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    features = []
    targets = []
    sample_ids = []

    for sample_info in batch:
        with open(sample_info["meta_path"], "r", encoding="utf-8") as f:
            meta = json.load(f)

        raw_data = meta.get("new_tokens_avg", None)
        if raw_data is None:
            raw_data = meta.get("new_tokens", None)

        if raw_data is None:
            tqdm.write(f"Warning: No target found in {sample_info['meta_path']}. Skipping.")
            continue

        probe_embedding = torch.load(sample_info["probe_path"], map_location="cpu")
        probe_embedding = probe_embedding.flatten().float()

        features.append(probe_embedding)
        targets.append(torch.tensor([float(raw_data)], dtype=torch.float32))
        sample_ids.append(sample_info["sample_id"])

    if not features:
        return None

    return torch.stack(features), torch.stack(targets), sample_ids


def collate_webdataset_fn(batch):
    features = []
    targets = []
    sample_ids = []

    for sample in batch:
        meta_raw = sample.get("json", None)
        feature_raw = sample.get("pth", None)
        if meta_raw is None or feature_raw is None:
            continue

        if isinstance(meta_raw, bytes):
            meta = json.loads(meta_raw.decode("utf-8"))
        elif isinstance(meta_raw, str):
            meta = json.loads(meta_raw)
        elif isinstance(meta_raw, dict):
            meta = meta_raw
        else:
            continue

        target = meta.get("target", None)
        if target is None:
            target = meta.get("new_tokens_avg", None)
        if target is None:
            target = meta.get("new_tokens", None)
        if target is None:
            continue

        sample_id = meta.get("sample_id", sample.get("__key__", ""))
        probe_embedding = torch.load(io.BytesIO(feature_raw), map_location="cpu")
        probe_embedding = probe_embedding.flatten().float()

        features.append(probe_embedding)
        targets.append(torch.tensor([float(target)], dtype=torch.float32))
        sample_ids.append(sample_id)

    if not features:
        return None

    return torch.stack(features), torch.stack(targets), sample_ids


def resolve_shard_urls(shard_patterns: str, arg_name: str) -> List[str]:
    """Expand comma-separated shard patterns to concrete tar paths."""
    patterns = [p.strip() for p in shard_patterns.split(",") if p.strip()]
    urls: List[str] = []

    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            urls.extend(matches)
            continue

        # No wildcard: treat as single path and keep explicit existence check.
        if not any(ch in pattern for ch in "*?[]") and Path(pattern).exists():
            urls.append(pattern)

    if not urls:
        raise FileNotFoundError(
            f"No shard files found for {arg_name}={shard_patterns!r}. "
            f"cwd={Path.cwd()}"
        )

    tqdm.write(f"Resolved {arg_name}: {len(urls)} shards")
    return urls


def infer_input_dim_from_webdataset(train_shards: List[str]) -> int:
    ds = wds.WebDataset(train_shards, shardshuffle=False)
    for sample in ds:
        feature_raw = sample.get("pth", None)
        if feature_raw is None:
            continue
        sample_h = torch.load(io.BytesIO(feature_raw), map_location="cpu")
        if sample_h.dim() > 1 and sample_h.shape[0] == 1:
            sample_h = sample_h.squeeze(0)
        return sample_h.flatten().shape[-1]
    raise RuntimeError(f"No valid sample found in train_shards: {train_shards[:3]} ...")


def build_webdataset_loader(
    shards: List[str],
    batch_size: int,
    pin_memory: bool,
    num_workers: int,
    prefetch_factor: int,
    shuffle_samples: bool,
    shuffle_buffer: int,
):
    ds = wds.WebDataset(shards, shardshuffle=shuffle_samples)
    if shuffle_samples:
        ds = ds.shuffle(shuffle_buffer)

    loader_kwargs = {
        "dataset": ds,
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": collate_webdataset_fn,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**loader_kwargs)


def evaluate_loader(model, data_loader, device, x_quantile: float = 0.9):
    model.eval()
    tracker = init_error_metric_tracker(x_quantile=x_quantile)

    with torch.no_grad():
        for batch in data_loader:
            if batch is None:
                continue
            features, targets, _sample_ids = batch
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            preds = model(features)

            abs_err = torch.abs(preds - targets)
            update_error_metric_tracker(tracker, abs_err)

            del features, targets, preds, abs_err

    return finalize_error_metric_tracker(tracker)


def collect_sample_mae(model, data_loader, device):
    model.eval()
    records = []

    with torch.no_grad():
        for batch in tqdm(data_loader, "testing"):
            if batch is None:
                continue
            features, targets, sample_ids = batch
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            preds = model(features)

            for sid, pred_t, true_t in zip(sample_ids, preds, targets):
                mae = abs(pred_t.item() - true_t.item())
                records.append((sid, mae, pred_t.item(), true_t.item()))

            del features, targets, preds

    records.sort(key=lambda x: x[1], reverse=True)
    return records


def write_sample_mae_jsonl(records, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        for sid, mae, pred, true in records:
            fp.write(
                json.dumps(
                    {
                        "sample_id": sid,
                        "mae": float(mae),
                        "pred": float(pred),
                        "target": float(true),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def train_and_evaluate():
    torch.manual_seed(42)
    args = get_args()

    data_dir = args.data_dir
    lr = 1e-4
    batch_size = 128
    epochs = 100
    runs = 1
    early_stop_patience = 15
    early_stop_warmup = 10
    num_workers = max(0, args.num_workers)
    prefetch_factor = max(1, args.prefetch_factor)

    test_maes: List[float] = []
    test_acc100s: List[float] = []
    test_acc200s: List[float] = []
    test_acc500s: List[float] = []
    test_accxp90s: List[float] = []
    test_xp90_values: List[float] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"Using device: {device}")
    pin_memory = torch.cuda.is_available()

    if args.data_backend == "webdataset":
        if wds is None:
            raise ImportError("webdataset is required for --data_backend webdataset")
        if not args.train_shards or not args.val_shards or not args.test_shards:
            raise ValueError("webdataset backend requires --train_shards --val_shards --test_shards")

        train_urls = resolve_shard_urls(args.train_shards, "--train_shards")
        val_urls = resolve_shard_urls(args.val_shards, "--val_shards")
        test_urls = resolve_shard_urls(args.test_shards, "--test_shards")

        input_dim = infer_input_dim_from_webdataset(train_urls)
        tqdm.write(f"Input Dim: {input_dim}")

        train_loader = build_webdataset_loader(
            shards=train_urls,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            shuffle_samples=True,
            shuffle_buffer=args.wds_shuffle_buffer,
        )
        val_loader = build_webdataset_loader(
            shards=val_urls,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            shuffle_samples=False,
            shuffle_buffer=args.wds_shuffle_buffer,
        )
        test_loader = build_webdataset_loader(
            shards=test_urls,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            shuffle_samples=False,
            shuffle_buffer=args.wds_shuffle_buffer,
        )
        test_samples_desc = "unknown(webdataset)"
    else:
        dataset = TraceDataset(data_dir)
        if len(dataset) == 0:
            return

        test_dataset_external = None
        if args.test_data_dir:
            test_dataset_external = TraceDataset(args.test_data_dir)
            if len(test_dataset_external) == 0:
                test_dataset_external = None

        sample_h = torch.load(dataset.samples[0]["probe_path"], map_location="cpu")
        if sample_h.dim() > 1 and sample_h.shape[0] == 1:
            sample_h = sample_h.squeeze(0)
        input_dim = sample_h.shape[-1]
        tqdm.write(f"Input Dim: {input_dim}")
        del sample_h

        if test_dataset_external is None:
            total_len = len(dataset)
            train_len = int(0.6 * total_len)
            val_len = int(0.2 * total_len)
            test_len = total_len - train_len - val_len
            tqdm.write(f"Data Split: Train={train_len}, Val={val_len}, Test={test_len}")
            train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])
        else:
            train_len = len(dataset)

            total_len = len(test_dataset_external)
            val_len = int(total_len * 0.1)
            test_len = int(total_len * 0.1)
            train2_len = total_len - val_len - test_len

            tqdm.write(f"Data Split: Train={train_len + train2_len}, Val={val_len}, Test={test_len} (external)")
            train2_dataset, val_dataset, test_dataset = random_split(test_dataset_external, [train2_len, val_len, test_len])
            train_dataset = ConcatDataset([dataset, train2_dataset])

        loader_common = {
            "batch_size": batch_size,
            "collate_fn": collate_fn,
            "pin_memory": pin_memory,
            "num_workers": num_workers,
            "persistent_workers": num_workers > 0,
        }
        if num_workers > 0:
            loader_common["prefetch_factor"] = prefetch_factor

        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **loader_common,
        )
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **loader_common,
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            **loader_common,
        )
        test_samples_desc = str(test_len)

    for run_idx in range(1, runs + 1):
        tqdm.write(f"\n=== Run {run_idx}/{runs} ===")

        model = LengthRegressorMLP(input_dim=input_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.SmoothL1Loss()

        best_val_mae = float("inf")
        best_model_path = Path(f"./pt_regression/best_model_run_{run_idx:02d}.pth")
        best_model_path.parent.mkdir(parents=True, exist_ok=True)

        no_improve_patience = 0

        tqdm.write("Start Training...")
        epoch_bar = tqdm(range(epochs), desc=f"Run {run_idx} Epochs", leave=False)
        for epoch in epoch_bar:
            model.train()
            train_loss_sum = 0.0
            train_mae_sum = 0.0
            samples_count = 0

            for batch in tqdm(train_loader, "training"):
                if batch is None:
                    continue
                features, targets, _sample_ids = batch
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                optimizer.zero_grad()
                preds = model(features)
                loss = loss_fn(preds, targets)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * targets.size(0)
                train_mae_sum += torch.abs(preds - targets).sum().item()
                samples_count += targets.size(0)

                del features, targets, preds, loss

            avg_train_loss = train_loss_sum / max(samples_count, 1)
            avg_train_mae = train_mae_sum / max(samples_count, 1)
            val_metrics = evaluate_loader(model, val_loader, device, x_quantile=0.9)
            val_mae = val_metrics["mae"]

            epoch_bar.set_postfix(
                {
                    "train_loss": f"{avg_train_loss:.2f}",
                    "train_mae": f"{avg_train_mae:.1f}",
                    "val_mae": f"{val_mae:.1f}",
                    "v@100": f"{val_metrics['acc100']*100:.1f}%",
                    "v@200": f"{val_metrics['acc200']*100:.1f}%",
                    "v@500": f"{val_metrics['acc500']*100:.1f}%",
                    "v@xp90": f"{val_metrics['accx']*100:.1f}%",
                },
                refresh=False,
            )

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(model.state_dict(), best_model_path)
                tqdm.write(f"  >>> Best Model Saved (Val MAE: {best_val_mae:.2f})")
                no_improve_patience = 0
            else:
                no_improve_patience += 1

            if epoch + 1 > early_stop_warmup and no_improve_patience >= early_stop_patience:
                tqdm.write(
                    "Early stop: val_mae did not improve for "
                    f"{early_stop_patience} epochs."
                )
                break

        tqdm.write("\n" + "=" * 50)
        tqdm.write("Training Finished. Starting COMPREHENSIVE EVALUATION on TEST SET...")
        tqdm.write("=" * 50)

        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            tqdm.write(f"Loaded best model from {best_model_path}")

            val_records = collect_sample_mae(model, val_loader, device)
            test_records = collect_sample_mae(model, test_loader, device)

            val_log_path = Path(f"./log_regression/best_model_run_{run_idx:02d}_val_sample_mae.tsv")
            test_log_path = Path(f"./log_regression/best_model_run_{run_idx:02d}_test_sample_mae.tsv")
            val_jsonl_path = Path(f"./log_regression/best_model_run_{run_idx:02d}_val_sample_mae.jsonl")
            test_jsonl_path = Path(f"./log_regression/best_model_run_{run_idx:02d}_test_sample_mae.jsonl")
            write_sample_mae_log(val_records, val_log_path)
            write_sample_mae_log(test_records, test_log_path)
            write_sample_mae_jsonl(val_records, val_jsonl_path)
            write_sample_mae_jsonl(test_records, test_jsonl_path)
            tqdm.write(f"Val sample MAE log saved to {val_log_path}")
            tqdm.write(f"Test sample MAE log saved to {test_log_path}")
            tqdm.write(f"Val sample MAE jsonl saved to {val_jsonl_path}")
            tqdm.write(f"Test sample MAE jsonl saved to {test_jsonl_path}")
        else:
            tqdm.write("Warning: No model saved. Using last epoch model.")

        test_metrics = evaluate_loader(model, test_loader, device, x_quantile=0.9)
        test_mae = test_metrics["mae"]
        test_acc100 = test_metrics["acc100"]
        test_acc200 = test_metrics["acc200"]
        test_acc500 = test_metrics["acc500"]
        test_accxp90 = test_metrics["accx"]
        test_xp90 = test_metrics["x_value"]
        tqdm.write("FINAL TEST REPORT:")
        tqdm.write(f"  Samples  : {test_samples_desc}")
        tqdm.write(f"  Test MAE : {test_mae:.2f} tokens")
        tqdm.write(f"  Acc@100  : {test_acc100 * 100:.2f}%")
        tqdm.write(f"  Acc@200  : {test_acc200 * 100:.2f}%")
        tqdm.write(f"  Acc@500  : {test_acc500 * 100:.2f}%")
        tqdm.write(f"  x(p90)   : {test_xp90:.2f} tokens")
        tqdm.write(f"  Acc@x(p90): {test_accxp90 * 100:.2f}%")
        tqdm.write("=" * 50)

        test_maes.append(test_mae)
        test_acc100s.append(test_acc100)
        test_acc200s.append(test_acc200)
        test_acc500s.append(test_acc500)
        test_accxp90s.append(test_accxp90)
        test_xp90_values.append(test_xp90)

    mae_mean = float(np.mean(test_maes))
    mae_var = float(np.var(test_maes))
    acc100_mean = float(np.mean(test_acc100s))
    acc100_var = float(np.var(test_acc100s))
    acc200_mean = float(np.mean(test_acc200s))
    acc200_var = float(np.var(test_acc200s))
    acc500_mean = float(np.mean(test_acc500s))
    acc500_var = float(np.var(test_acc500s))
    accxp90_mean = float(np.mean(test_accxp90s))
    accxp90_var = float(np.var(test_accxp90s))
    xp90_mean = float(np.mean(test_xp90_values))
    xp90_var = float(np.var(test_xp90_values))

    tqdm.write("\n=== Summary over runs ===")
    tqdm.write(f"MAE mean: {mae_mean:.2f} | variance: {mae_var:.6f}")
    tqdm.write(f"Acc@100 mean: {acc100_mean * 100:.2f}% | variance: {acc100_var * 100:.6f}")
    tqdm.write(f"Acc@200 mean: {acc200_mean * 100:.2f}% | variance: {acc200_var * 100:.6f}")
    tqdm.write(f"Acc@500 mean: {acc500_mean * 100:.2f}% | variance: {acc500_var * 100:.6f}")
    tqdm.write(f"x(p90) mean: {xp90_mean:.2f} | variance: {xp90_var:.6f}")
    tqdm.write(f"Acc@x(p90) mean: {accxp90_mean * 100:.2f}% | variance: {accxp90_var * 100:.6f}")


if __name__ == "__main__":
    train_and_evaluate()
