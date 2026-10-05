"""
Hyperparameters and the MAINRUN_CONFIG override loader.

Imported by train.py. Deliberately free of torch/utils imports so it is
testable outside the devcontainer. No `from __future__ import annotations`
here on purpose - we read field types off the dataclass at runtime.
"""
import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Hyperparameters:
    block_size: int = 128
    batch_size: int = 64
    vocab_size: int = 16_000
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 512
    dropout: float = 0.1
    lr: float = 6e-3
    weight_decay: float = 0.0
    evals_per_epoch: int = 3

    epochs: int = 7
    seed: int = 1337
    num_titles: int = 100_000
    val_frac: float = 0.10
    log_file: str = "./logs/mainrun.log"

    run_label: str = ""

    # --- section 4.1 AdamW (ignored unless optimizer == "adamw") ---
    optimizer: str = "sgd"
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_frac: float = 0.06
    lr_floor_frac: float = 0.1

    # --- section 4.5 scaled residual init (GPT-2 depth-scaled std on the two
    #     per-block residual-path projections; ignored when False) ---
    scaled_residual_init: bool = False

    # --- section 4.2 tokenizer training corpus ---
    tokenizer_scope: str = "train_and_val"

    # --- section 4.6 batch shuffling (chunk-shuffle vs sequential pointer walk) ---
    shuffle_batches: bool = False

    # --- section 4.11 / 4.12 deprioritised toggles ---
    attn_impl: str = "manual"          # "manual" | "sdpa"
    grad_clip_norm: float = 1.0        # was hard-coded at 1.0
    use_amp: bool = False              # bf16 autocast on the fwd/bwd
    lr_schedule: str = "cosine"        # "cosine" | "wsd" (adamw / muon paths)
    stable_frac: float = 0.6           # wsd: hold at peak lr for this frac of post-warmup steps

    # --- section 4.1-follow-up Muon (ignored unless optimizer == "muon";
    #     the non-Muon params run on AdamW using lr/beta1/beta2/weight_decay) ---
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5

    # --- eval-time weight EMA (0.0 = disabled; pure eval transform) ---
    ema_decay: float = 0.0

    # --- Phase 11 (Session B) ---
    weight_decay_embeddings: bool = True   # False -> no decay on token_emb/pos_emb either
    mlp_kind: str = "gelu"                 # "gelu" | "swiglu" (param-matched)
    qk_norm: bool = False                  # RMSNorm on q,k before attention
    wsd_decay_shape: str = "linear"        # "linear" | "invsqrt" (wsd tail)


# Fixed for the whole exercise (spec section 1). The loader rejects any override.
PROTECTED_FIELDS = frozenset({"epochs", "seed", "num_titles", "val_frac", "log_file"})

# Fields whose value must come from a fixed set (spec Literal[...] fields).
_CHOICES = {
    "optimizer": frozenset({"sgd", "adamw", "muon"}),
    "tokenizer_scope": frozenset({"train_and_val", "train_only"}),
    "attn_impl": frozenset({"manual", "sdpa"}),
    "lr_schedule": frozenset({"cosine", "wsd"}),
    "mlp_kind": frozenset({"gelu", "swiglu"}),
    "wsd_decay_shape": frozenset({"linear", "invsqrt"}),
}


class ConfigError(ValueError):
    """Raised for any malformed or disallowed MAINRUN_CONFIG content."""


def load_config(path: Optional[str]) -> Hyperparameters:
    """No path -> exact defaults. Otherwise apply the JSON object's keys over
    Hyperparameters(), rejecting unknown keys, protected keys, and wrong types.

    Validation order per key: unknown -> protected -> type. Field types are read
    from the default instance (type(getattr(defaults, key))), which avoids the
    stringified-annotation problem and needs no typing introspection.
    """
    if path is None:
        return Hyperparameters()

    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ConfigError(
            f"{path}: top-level JSON must be an object, got {type(raw).__name__}"
        )

    defaults = Hyperparameters()
    valid = {f.name for f in dataclasses.fields(Hyperparameters)}
    for key, value in raw.items():
        if key not in valid:
            raise ConfigError(
                f"unknown hyperparameter {key!r} (not a field of Hyperparameters)"
            )
        if key in PROTECTED_FIELDS:
            raise ConfigError(
                f"{key!r} is fixed for this exercise and cannot be overridden"
            )
        expected = type(getattr(defaults, key))
        if expected is bool:
            ok = isinstance(value, bool)
        elif expected is int:
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif expected is float:
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif expected is str:
            ok = isinstance(value, str)
        else:
            ok = isinstance(value, expected)
        if not ok:
            raise ConfigError(
                f"{key!r} must be {expected.__name__}, got {type(value).__name__}"
            )
        if key in _CHOICES and value not in _CHOICES[key]:
            allowed = ", ".join(sorted(_CHOICES[key]))
            raise ConfigError(
                f"{key!r} must be one of {{{allowed}}}, got {value!r}"
            )

    return dataclasses.replace(defaults, **raw)


def warmup_cosine_factor(step, max_steps, warmup_frac, lr_floor_frac):
    """LR multiplier for training-step index `step` (0-based, < max_steps).

    Linear ramp from 1/W to 1.0 over the first W = round(warmup_frac*max_steps)
    steps, then a half-cosine from 1.0 down to lr_floor_frac over the rest.
    """
    warmup = round(warmup_frac * max_steps)
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    denom = max(1, max_steps - warmup)
    progress = min(1.0, max(0.0, (step - warmup) / denom))
    return lr_floor_frac + (1.0 - lr_floor_frac) * 0.5 * (1.0 + math.cos(math.pi * progress))


def wsd_factor(step, max_steps, warmup_frac, lr_floor_frac, stable_frac,
               shape="linear"):
    """§4.12 warmup-stable-decay LR multiplier: linear warmup to 1.0, hold at
    1.0 for `stable_frac` of the post-warmup steps, then decay to
    `lr_floor_frac` over the remainder. `shape` picks the decay curve:
    "linear" (default) or "invsqrt" (1-sqrt, drops sooner then flattens)."""
    warmup = round(warmup_frac * max_steps)
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    post = max(1, max_steps - warmup)
    stable = round(stable_frac * post)
    if step < warmup + stable:
        return 1.0
    progress = min(1.0, max(0.0, (step - warmup - stable) / max(1, post - stable)))
    if shape == "invsqrt":
        # "1-sqrt" decay: lr/lr_max = 1 - (1-floor)*sqrt(p) -> drops fast early,
        # flattens near the floor. (MiniCPM / WSD scaling-law follow-ups.)
        progress = progress ** 0.5
    return 1.0 + (lr_floor_frac - 1.0) * progress
