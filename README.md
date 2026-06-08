# DecodingPredict: Prefill-Based Decoding Length Prediction

This repository contains the core code for our paper on decoding length prediction from prefill-phase hidden states, including theoretical analysis of prediction error bounds.

## Structure

```
paper_code_release/
├── data_generation/       # Export prefill hidden states & decode metadata
│   └── utils.py           # Shared utilities for model loading, KV cache handling, etc.
├── prediction/            # Predictor training and evaluation
│   ├── processors.py      # Feature extraction from hidden states (mean/last/concat)
│   ├── model_regression.py # Lightweight MLP regressor
│   ├── train_regression.py # Training script with pass_rate metrics
│   └── train_utils.py     # Shared training utilities
├── analysis/              # Statistical analysis scripts
│   ├── analyze_shapiro_per_sample.py  # Normality testing (Shapiro-Wilk)
│   ├── analyze_test_mae_vs_std.py     # Correlation between MAE and σ
│   └── analyze_mean_std_correlation.py # μ–σ relationship across models
└── requirements.txt
```

## Quick Start

### 1. Environment

```bash
conda create -n decodingpredict python=3.12 -y
conda activate decodingpredict
pip install -r requirements.txt
```

### 2. Data Generation

Export prefill hidden states, KV cache, attention maps, and decode metadata:

```bash
python -m data_generation.utils --help
```

The main export scripts (e.g., for ShareGPT, HumanEval, GSM8K) follow the same pattern:
- Load target LLM with `attn_implementation="eager"`
- Run prefill, save `prefill_hidden_states.pt` and `prefill_kv_cache.pt`
- Run decoding, save `decode_metadata.json` with `new_tokens`

### 3. Training

Train the baseline MLP regressor on concatenated hidden states:

```bash
cd prediction
python train_regression.py \
  --data-dir ../../data/<model+thinking>/<dataset>_outputs/<split> \
  --hidden-types concat \
  --epochs 100 \
  --output-dir ./log_regression
```

Key arguments:
- `--hidden-types`: choose from `average`, `last_token`, `concat`, `entropy`
- `--layer-mode`: `mean` or `last` for layer aggregation
- `--token-mode`: `mean` for token aggregation

### 4. Evaluation

Metrics reported: MAE, pass_rate@100/200/500/1000, and x(p90) (the smallest X such that pass_rate@X ≥ 90%).

### 5. Analysis

Compute theoretical bounds and validate normality:

```bash
cd analysis
# Per-sample Shapiro-Wilk test
python analyze_shapiro_per_sample.py --input path/to/per_sample_stats.jsonl

# MAE vs σ correlation
python analyze_test_mae_vs_std.py \
  --test-mae-path path/to/test_sample_mae.jsonl \
  --std-path path/to/per_sample_stats.jsonl

# μ–σ model characterization
python analyze_mean_std_correlation.py --input path/to/per_sample_stats.jsonl
```

## Citation

If you use this code, please cite our paper:

```bibtex
@article{decodingpredict2026,
  title={On the Fundamental Limits of Prefill-Based Decoding Length Prediction},
  author={[Author Name]},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

MIT
