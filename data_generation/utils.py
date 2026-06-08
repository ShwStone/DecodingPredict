from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.tokenization_utils_base import BatchEncoding

# ----------------------------
# Configuration
# ----------------------------

MAX_NEW_TOKENS: int = 8192
ChatMessage = Dict[str, str]

# ----------------------------
# Type aliases (for readability)
# ----------------------------

TensorDict = Dict[str, torch.Tensor]
HiddenStates = Tuple[torch.Tensor, ...]
PastKeyValues = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
ProbeEmbedding = Optional[torch.Tensor]


def sanitize_model_identifier(model_id: str) -> str:
    """Create a filesystem-safe slug for a model identifier."""
    import re

    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id.strip())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "model"


def describe_thinking_suffix(thinking_flag: Optional[bool], thinking_mode: str) -> str:
    if thinking_flag is True:
        return "with-thinking"
    if thinking_flag is False:
        return "no-thinking"
    return f"{thinking_mode}-thinking"


def build_output_root(
    base_dir: Path,
    model_id: str,
    thinking_flag: Optional[bool],
    thinking_mode: str,
    output_subdir: str,
) -> Path:
    model_slug = sanitize_model_identifier(model_id)
    thinking_suffix = describe_thinking_suffix(thinking_flag, thinking_mode)
    run_segment = f"{model_slug}+{thinking_suffix}"
    return base_dir / run_segment / output_subdir


def prefill_artifact_paths(
    sample_dir: Path,
    probe_prompt: Optional[str],
    hidden_types: Iterable[str],
) -> List[Path]:
    hidden = set(hidden_types)
    required: List[Path] = []
    if "entropy" in hidden:
        required.append(sample_dir / "prefill" / "prefill_entropy_pooled_hidden.pt")
    if "average" in hidden:
        required.append(sample_dir / "prefill" / "prefill_average_pooled_hidden.pt")
    if "last_token" in hidden:
        required.append(sample_dir / "prefill" / "prefill_last_token_hidden.pt")
    if "concat" in hidden:
        required.append(sample_dir / "prefill" / "prefill_concat_last_token_hidden.pt")
    if probe_prompt and "probe" in hidden:
        required.append(sample_dir / "prefill" / "prefill_probe_embedding.pt")
    if "echo" in hidden:
        required.append(sample_dir / "prefill" / "prefill_probe_echo_embedding.pt")
    return required


def prefill_artifacts_exist(
    sample_dir: Path,
    probe_prompt: Optional[str],
    hidden_types: Iterable[str],
) -> bool:
    required = prefill_artifact_paths(sample_dir, probe_prompt, hidden_types)
    return all(path.exists() for path in required)


def save_prefill_artifacts(
    target_dir: Path,
    probe_embedding: Optional[torch.Tensor],
    echo_embedding: Optional[torch.Tensor],
    entropy_pooled_hidden: Optional[torch.Tensor],
    last_token_hidden: Optional[torch.Tensor],
    hidden_types: Iterable[str],
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    hidden = set(hidden_types)

    if "probe" in hidden and probe_embedding is not None:
        torch.save(probe_embedding, target_dir / "prefill_probe_embedding.pt")
    if "echo" in hidden and echo_embedding is not None:
        torch.save(echo_embedding, target_dir / "prefill_probe_echo_embedding.pt")
    if "entropy" in hidden and entropy_pooled_hidden is not None:
        torch.save(
            entropy_pooled_hidden,
            target_dir / "prefill_entropy_pooled_hidden.pt",
        )
    if "last_token" in hidden and last_token_hidden is not None:
        torch.save(
            last_token_hidden,
            target_dir / "prefill_last_token_hidden.pt",
        )

def render_chat_prompt(
    tokenizer: AutoTokenizer,
    template: list[ChatMessage],
    enable_thinking: Optional[bool] = None,
) -> str:
    extra_kwargs = {}
    if enable_thinking is not None:
        extra_kwargs["enable_thinking"] = enable_thinking

    prompt = tokenizer.apply_chat_template(
        template,
        tokenize=False,
        add_generation_prompt=True,
        **extra_kwargs,
    )
    return prompt


def ensure_pad_token(tokenizer: AutoTokenizer) -> AutoTokenizer:
    """Ensure tokenizer has a PAD token. If missing, reuse EOS."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def ensure_attention_mask(batch: TensorDict) -> TensorDict:
    """Ensure `attention_mask` exists; default to all-ones."""
    if "attention_mask" not in batch:
        batch["attention_mask"] = torch.ones_like(batch["input_ids"])
    return batch


def to_device(batch: TensorDict, device: torch.device) -> TensorDict:
    """Move all tensors in a tokenized batch to a device."""
    return {key: value.to(device) for key, value in batch.items()}


def summarize_hidden_states(
    hidden_states: HiddenStates, max_layers: int = 4
) -> List[str]:
    """Summarize the first few hidden-state tensors (shape, dtype, mean/std)."""
    total_layers: int = len(hidden_states)
    lines: List[str] = []

    for idx, layer_hidden in enumerate(hidden_states[:max_layers]):
        shape = tuple(layer_hidden.shape)
        mean_val = layer_hidden.mean().item()
        std_val = layer_hidden.std().item()
        lines.append(
            f"layer {idx:02d}/{total_layers - 1:02d} -> "
            f"shape={shape} dtype={layer_hidden.dtype} mean={mean_val:.4f} std={std_val:.4f}"
        )

    if total_layers > max_layers:
        lines.append(f"... ({total_layers - max_layers} more layers)")
    return lines


def run_prefill(
    model: torch.nn.Module,
    batch: TensorDict,
    output_hidden_states: bool = False,
    use_cache: bool = True,
) -> CausalLMOutputWithPast:
    """Run prefill forward pass and request KV cache (optionally hidden states)."""
    with torch.no_grad():
        return model(
            **batch,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            return_dict=True,
        )


def run_generation(model: torch.nn.Module, tokenizer: AutoTokenizer, batch: TensorDict):
    """Run normal random generation while returning the number of output tokens and results string."""
    with torch.no_grad():
        generation = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )
    return generation


def encode(
    tokenizer: AutoTokenizer,
    chat_template: list[ChatMessage],
    device: torch.device,
    enable_thinking: Optional[bool] = None,
) -> TensorDict:
    """Tokenize text and move to device."""
    prompt = render_chat_prompt(tokenizer, chat_template, enable_thinking=enable_thinking)
    enc: BatchEncoding = tokenizer(prompt, return_tensors="pt")
    batch: TensorDict = ensure_attention_mask(dict(enc))
    batch = to_device(batch, device)
    return batch

def _build_probe_tokens(
    tokenizer: AutoTokenizer,
    probe_prompt: Optional[str],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not probe_prompt:
        return None

    probe_tokens = tokenizer(
        probe_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)
    if probe_tokens.numel() == 0:
        return None
    return probe_tokens


def run_probe_embedding(
    model: torch.nn.Module,
    batch: TensorDict,
    prefill: CausalLMOutputWithPast,
    tokenizer: AutoTokenizer,
    probe_prompt: Optional[str],
) -> ProbeEmbedding:
    """Append probe prompt after chat template and return last-token hidden state."""
    probe_tokens = _build_probe_tokens(tokenizer, probe_prompt, model.device)
    if probe_tokens is None:
        return None

    attention_mask = batch["attention_mask"]
    batch_size = attention_mask.shape[0]
    probe_len = probe_tokens.shape[1]
    ones = torch.ones(
        (batch_size, probe_len),
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    extended_attention_mask = torch.cat([attention_mask, ones], dim=-1)
    prefill_len = attention_mask.sum(dim=-1, keepdim=True)
    position_ids = prefill_len + torch.arange(
        probe_len, device=attention_mask.device
    ).unsqueeze(0)

    with torch.no_grad():
        probe_out = model(
            input_ids=probe_tokens,
            attention_mask=extended_attention_mask,
            position_ids=position_ids,
            past_key_values=prefill.past_key_values,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    embedding = probe_out.hidden_states[-1][:, -1, :].detach().cpu()
    return embedding.squeeze(0)


def entropy_weighted_pooling(
    last_hidden: torch.Tensor,
    logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pool last-layer hidden states using entropy-based token weights.

    Weight for token i uses entropy from logits at position i-1, since the
    previous token predicts the current one.
    """
    logits = logits.float()
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)  # [B, L]

    # shift entropy right: weight for token i comes from entropy at i-1
    entropy = torch.nn.functional.pad(entropy[:, :-1], (1, 0), value=entropy.min().item())

    if attention_mask is not None:
        mask = attention_mask.to(dtype=entropy.dtype)
        entropy = entropy.masked_fill(mask == 0, float("-inf"))

    weights = torch.nn.functional.softmax(entropy, dim=-1)  # [B, L]
    pooled = (weights.unsqueeze(-1) * last_hidden).sum(dim=1)
    return pooled.detach()


def average_pooling(
    last_hidden: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Average-pool last-layer hidden states over tokens."""
    if attention_mask is None:
        return last_hidden.mean(dim=1).detach()

    mask = attention_mask.to(dtype=last_hidden.dtype).unsqueeze(-1)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    pooled = summed / counts
    return pooled.detach()


def concat_last_token_all_layers(hidden_states: HiddenStates) -> torch.Tensor:
    """Concatenate the last-token hidden state from all layers."""
    if not hidden_states:
        raise ValueError("hidden_states is empty")
    last_tokens = [layer[:, -1, :].detach() for layer in hidden_states]
    return torch.cat(last_tokens, dim=-1).cpu()


def gen_data(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    chat_template: list[ChatMessage],
    enable_thinking: Optional[bool] = None,
    probe_prompt: Optional[str] = None,
    output_hidden_states: bool = False,
) -> Tuple[CausalLMOutputWithPast, CausalLMOutputWithPast, ProbeEmbedding]:
    """
    Generate data: prefill, generation outputs, and optional probe embedding.
    """
    torch.set_grad_enabled(False)

    tokenizer = ensure_pad_token(tokenizer)
    model.eval()  # Important: disable dropout for deterministic cache verification.

    # Tokenize prompt.
    batch = encode(tokenizer, chat_template, model.device, enable_thinking=enable_thinking)

    # 1) Prefill stage
    prefill = run_prefill(model, batch, output_hidden_states=output_hidden_states)

    # 2) Probe embedding (append after chat template)
    probe_embedding = run_probe_embedding(
        model,
        batch,
        prefill,
        tokenizer,
        probe_prompt,
    )

    # 3) Generation stage
    generation = run_generation(model, tokenizer, batch)

    return prefill, generation, probe_embedding


def main() -> None:
    torch.set_grad_enabled(False)
    MODEL_ID: str = "Qwen/Qwen2.5-0.5B-Instruct"
    CHAT_TEMPLATE: list[ChatMessage] = [
        {
            "role": "system",
            "content": "You are a concise assistant that stops after answering.",
        },
        {"role": "user", "content": "Tell me, what 1 + 1 equals to?"},
    ]
    PROBE_PROMPT: str = (
        "\nBased on the difficulty of providing an answer, predict the expected length of the next response (choose from \"very short\", \"short\", \"medium\", or \"long\", \"very long\"): \""
    )

    tokenizer = ensure_pad_token(AutoTokenizer.from_pretrained(MODEL_ID, padding_side='left'))
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype="auto"
    )
    model.eval()

    # Base prompt
    prompt = render_chat_prompt(tokenizer, CHAT_TEMPLATE)
    enc: BatchEncoding = tokenizer(prompt, return_tensors="pt")
    batch: TensorDict = ensure_attention_mask(dict(enc))
    batch = to_device(batch, model.device)

    # Prefill stage
    print("Running prefill stage...")
    prefill = run_prefill(model, batch)

    # Prefill token weights (entropy-shifted)
    with torch.no_grad():
        logits = prefill.logits.float()
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)  # [B, L]
        entropy = torch.nn.functional.pad(
            entropy[:, :-1], (1, 0), value=entropy.min().item()
        )
        mask = batch["attention_mask"].to(dtype=entropy.dtype)
        entropy = entropy.masked_fill(mask == 0, float("-inf"))
        weights = torch.nn.functional.softmax(entropy, dim=-1)

    input_ids = batch["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    weight_values = weights[0].tolist()
    mask_values = batch["attention_mask"][0].tolist()

    print("\n--- Prefill token weights ---")
    for token, weight, m in zip(tokens, weight_values, mask_values):
        if m == 0:
            continue
        print(f"{token}\t{weight:.6f}")

    # Cached probe logits
    probe_tokens = _build_probe_tokens(tokenizer, PROBE_PROMPT, model.device)
    if probe_tokens is None:
        raise RuntimeError("Probe prompt tokenization failed.")

    attention_mask = batch["attention_mask"]
    probe_len = probe_tokens.shape[1]
    ones = torch.ones(
        (attention_mask.shape[0], probe_len),
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    extended_attention_mask = torch.cat([attention_mask, ones], dim=-1)
    prefill_len = attention_mask.sum(dim=-1, keepdim=True)
    position_ids = prefill_len + torch.arange(
        probe_len, device=attention_mask.device
    ).unsqueeze(0)

    with torch.no_grad():
        cached_probe_out = model(
            input_ids=probe_tokens,
            attention_mask=extended_attention_mask,
            position_ids=position_ids,
            past_key_values=prefill.past_key_values,
            use_cache=False,
            return_dict=True,
        )

    cached_logits = cached_probe_out.logits[:, -1, :].float()

    # Full prompt + probe logits (no cache)
    full_prompt = prompt + PROBE_PROMPT
    full_enc: BatchEncoding = tokenizer(full_prompt, return_tensors="pt")
    full_batch: TensorDict = ensure_attention_mask(dict(full_enc))
    full_batch = to_device(full_batch, model.device)

    with torch.no_grad():
        full_out = model(
            **full_batch,
            use_cache=False,
            return_dict=True,
        )

    full_logits = full_out.logits[:, -1, :].float()

    # KL divergence between cached vs full logits
    cached_log_probs = torch.nn.functional.log_softmax(cached_logits, dim=-1)
    full_probs = torch.nn.functional.softmax(full_logits, dim=-1)
    kl_div = torch.nn.functional.kl_div(
        cached_log_probs, full_probs, reduction="batchmean"
    )

    print("\n--- Probe KV cache verification ---")
    print(f"KL divergence (cached vs full): {kl_div.item():.6e}")


if __name__ == "__main__":
    main()
