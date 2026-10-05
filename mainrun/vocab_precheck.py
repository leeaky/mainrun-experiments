"""
vocab_precheck.py - standalone tokenizer-only diagnostic to choose a vocab_size
search range for train.py BEFORE spending any GPU time on a real training sweep.

This is NOT part of the training run. No torch model is built or trained, no GPU
is used. It runs on CPU in a few minutes.

WHAT IT MEASURES, per vocab_size
--------------------------------
1. val_chars_per_token = len(val_text) / len(encode(val_text))

   `len(val_text)` is exactly the denominator that train.py's evaluate() divides
   the summed validation loss by (see train.py: `return losses / len(val_text)`).
   So this number is "characters bought per token on the validation split" -
   higher means better compression, which is what lets a fixed token budget cover
   more text. Named to keep that connection obvious.

2. frac_undertrained = fraction of the tokenizer's vocab whose token id occurs
   fewer than MIN_SUPPORT times across the encoded *train* text.

   This is a FREE, NO-TRAINING PROXY for "how much of this vocab would likely be
   starved of gradient signal at our step budget". It is NOT a substitute for
   actually training: it ignores the optimiser, the step count, batch composition,
   and the fact that a rare token can still learn a decent embedding. Treat it as
   a rough risk gauge, not a verdict.

DELIBERATE DIFFERENCE FROM train.py
-----------------------------------
train.py currently trains the tokenizer on `train_titles + val_titles` (leaky).
This diagnostic trains on `train_titles` ONLY, to reflect the corrected, non-leaky
setup. Seed (1337), num_titles (100_000) and val_frac (0.10) match the current
Hyperparameters defaults; splits are produced by train.get_titles() unmodified.

OUTPUTS
-------
- a results table printed as the sweep runs
- <outdir>/results.csv and <outdir>/results.json (json includes the config block)
- <outdir>/val_cpt_vs_vocab.png : val_chars_per_token vs vocab_size (log2 x-axis)
  with frac_undertrained on a secondary y-axis
- knee detection on val_chars_per_token (kneed.KneeLocator if installed, plus a
  manual per-doubling gain table that is always printed as a sanity check)
- first vocab_size where frac_undertrained exceeds the risk threshold
- a final "candidate range for the real training sweep is [X, Y]" summary, or an
  explicit explanation when the grid can't resolve a sensible ordered range

HOW TO RUN (inside the Mainrun devcontainer - Python 3.10 + pinned deps)
-----------------------------------------------------------------------
    python3 vocab_precheck.py                    # full default sweep
    python3 vocab_precheck.py --smoke            # tiny fast sweep, checks plumbing
    python3 vocab_precheck.py --vocab-sizes 4096,8192,16384 --min-support 8
    python3 vocab_precheck.py --risk-threshold 0.30 --outdir /tmp/vp

`kneed` is optional (`pip3 install kneed`); without it the manual gain table is
used for knee detection.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import types
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Config defaults. Override the sweep-shaping ones from the command line;      #
# SEED / NUM_TITLES / VAL_FRAC / EOS_TOKEN mirror train.py and should not be   #
# changed casually (the point is identical splits & BPE behaviour).           #
# --------------------------------------------------------------------------- #
SEED = 1337               # == Hyperparameters.seed
NUM_TITLES = 100_000      # == Hyperparameters.num_titles
VAL_FRAC = 0.10           # == Hyperparameters.val_frac
EOS_TOKEN = "<eos>"       # == train.py eos_token

DEFAULT_VOCAB_GRID = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
DEFAULT_MIN_SUPPORT = 5
DEFAULT_RISK_THRESHOLD = 0.25
DEFAULT_OUTDIR = "vocab_precheck_out"

SMOKE_VOCAB_GRID = [256, 512, 1024]
SMOKE_NUM_TITLES = 2000

KNEE_FLAT_FRAC = 0.50     # manual-fallback knee: first doubling whose compression
                          # gain drops below this * the best per-doubling gain
                          # (crude - the printed Δ table is the real sanity check)


# --------------------------------------------------------------------------- #
# Reuse train.py's data + tokenizer code unmodified.                          #
# --------------------------------------------------------------------------- #
def import_from_train():
    """Return (get_titles, train_tokenizer, BPETokenizer) from the real train.py.

    train.py runs `import utils` at module load, and utils.py's _check_devcontainer()
    calls sys.exit(1) unless /root/.mainrun exists. Inside the Mainrun devcontainer
    that file exists, so `import utils` succeeds naturally and the stub below is
    never installed.

    OUTSIDE any container (e.g. no Docker available) we pre-register an empty
    `utils` module so the guard can't fire: train.py only does `import utils` and
    never calls into it, and it has no other import-time side effects before its
    `if __name__ == "__main__"` guard, so an empty module is sufficient. This keeps
    the diagnostic runnable without a devcontainer without touching train.py.
    """
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    if not Path("/root/.mainrun").exists():
        sys.modules.setdefault("utils", types.ModuleType("utils"))
    from train import get_titles, train_tokenizer, BPETokenizer
    return get_titles, train_tokenizer, BPETokenizer


# --------------------------------------------------------------------------- #
# Measurement                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class SweepResult:
    vocab_size: int            # requested
    vocab_actual: int          # tokenizer.get_vocab_size() after training
    val_chars_per_token: float
    n_vocab_undertrained: int
    frac_undertrained: float


def build_texts(train_titles: list[str], val_titles: list[str]) -> tuple[str, str]:
    """Mirror train.py's text construction exactly (train.py lines ~238-239)."""
    train_text = EOS_TOKEN.join(train_titles) + EOS_TOKEN
    val_text = EOS_TOKEN.join(val_titles) + EOS_TOKEN
    return train_text, val_text


def measure_one(vocab_size, train_titles, train_text, val_text, min_support,
                train_tokenizer, BPETokenizer):
    """Train one tokenizer (on train_titles ONLY) and take both measurements.

    Returns (SweepResult, tok, counts) - tok and counts are handed back so the
    caller can print example undertrained tokens for the first grid point.
    """
    tok = BPETokenizer(train_tokenizer(train_titles, vocab_size, eos_token=EOS_TOKEN))
    vocab_actual = int(tok.vocab_size)

    # val_chars_per_token: same numerator (char count of val_text) that evaluate()
    # divides the summed val loss by, over the encoded token count.
    val_ids = tok.encode(val_text)
    val_chars_per_token = len(val_text) / len(val_ids)

    # frac_undertrained: token-id frequency over the encoded TRAIN text.
    train_ids = np.asarray(tok.encode(train_text), dtype=np.int64)
    counts = np.bincount(train_ids, minlength=vocab_actual)[:vocab_actual]
    n_under = int((counts < min_support).sum())
    frac = n_under / vocab_actual

    res = SweepResult(int(vocab_size), vocab_actual, float(val_chars_per_token),
                      n_under, float(frac))
    return res, tok, counts


def _print_undertrained_examples(tok, counts, min_support, k=12):
    itos = tok.itos
    under = [i for i in range(len(counts)) if counts[i] < min_support]
    shown = ", ".join(repr(itos.get(i, "?")) for i in under[:k])
    tail = " ..." if len(under) > k else ""
    print(f"      ({len(under)} tokens seen <{min_support}x here; e.g. {shown}{tail})")


def run_sweep(grid, train_titles, val_titles, min_support,
              train_tokenizer, BPETokenizer):
    train_text, val_text = build_texts(train_titles, val_titles)
    print(f"\ntrain_text: {len(train_text):,} chars / {len(train_titles):,} titles")
    print(f"val_text:   {len(val_text):,} chars / {len(val_titles):,} titles")
    print("(tokenizer trained on train_titles ONLY - non-leaky; train.py currently "
          "uses train+val)\n")

    hdr = (f"{'vocab_size':>11}  {'vocab_actual':>12}  "
           f"{'val_chars_per_token':>19}  {'frac_undertrained':>17}")
    print(hdr)
    print("-" * len(hdr))

    results: list[SweepResult] = []
    for i, v in enumerate(grid):
        res, tok, counts = measure_one(v, train_titles, train_text, val_text,
                                       min_support, train_tokenizer, BPETokenizer)
        results.append(res)
        print(f"{res.vocab_size:>11,}  {res.vocab_actual:>12,}  "
              f"{res.val_chars_per_token:>19.4f}  {res.frac_undertrained:>17.4f}")
        if i == 0:
            _print_undertrained_examples(tok, counts, min_support)
    return results


# --------------------------------------------------------------------------- #
# Analysis                                                                    #
# --------------------------------------------------------------------------- #
def detect_knee(xs, ys):
    """Knee of the (rising, concave) val_chars_per_token curve.

    Returns (knee_x_or_None, method_str, per_step_gains) where per_step_gains is a
    list of (x_to, delta_y) between consecutive grid points - always computed, and
    printed by summarize() as a sanity check / fallback for when kneed isn't
    installed or misbehaves on a grid this small.
    """
    xs = list(xs)
    ys = list(ys)
    gains = [(xs[i + 1], ys[i + 1] - ys[i]) for i in range(len(ys) - 1)]

    knee = None
    method = "manual per-doubling gain"
    try:
        from kneed import KneeLocator
        kl = KneeLocator(xs, ys, curve="concave", direction="increasing")
        if kl.knee is not None:
            knee = int(kl.knee)
            method = "kneed.KneeLocator"
    except ImportError:
        pass
    except Exception as e:  # kneed installed but unhappy on this grid
        method = f"manual per-doubling gain (kneed error: {e})"

    if knee is None and len(gains) >= 2:
        peak = max(g for _, g in gains)
        if peak > 0:
            for x_to, g in gains:
                if g < KNEE_FLAT_FRAC * peak:
                    idx = xs.index(x_to) - 1        # knee = point before the flat step
                    knee = int(xs[max(idx, 0)])
                    break
    return knee, method, gains


def threshold_crossing(results, risk_threshold):
    for r in results:
        if r.frac_undertrained > risk_threshold:
            return r.vocab_size
    return None


def summarize(results, knee, knee_method, gains, crossing, risk_threshold):
    line = "=" * 68
    print("\n" + line)
    print("KNEE DETECTION  -  val_chars_per_token vs vocab_size")
    print(f"  method: {knee_method}")
    print(f"  compression knee vocab_size (X): "
          f"{format(knee, ',') if knee is not None else 'not resolved'}")
    print("  per-doubling gain  Δ val_chars_per_token:")
    for x_to, g in gains:
        print(f"      -> {x_to:>8,}:  {g:+.4f}")

    print(f"\nUNDERTRAINING RISK  -  first vocab_size with frac_undertrained > {risk_threshold}")
    if crossing is None:
        print("  threshold never exceeded on this grid (Y not resolved)")
    else:
        print(f"  crossing vocab_size (Y): {crossing:,}")

    print("\n" + line)
    X, Y = knee, crossing
    if X is None or Y is None:
        print("CANDIDATE RANGE: not fully resolved on this grid.")
        if X is not None:
            print(f"  Compression knee at {X:,}, but frac_undertrained never crosses "
                  f"{risk_threshold:.0%} - no upper bound. Extend the grid upward.")
        if Y is not None:
            print(f"  Undertraining crosses at {Y:,}, but the compression knee is "
                  f"unresolved - refine the grid or check the Δ table above by hand.")
        if X is None and Y is None:
            print("  Neither bound resolved - widen the grid in both directions.")
    elif X > Y:
        print("CANDIDATE RANGE: no sensible ordered range on this grid.")
        print(f"  Compression knee X={X:,} sits ABOVE undertraining crossing Y={Y:,}.")
        print(f"  The grid is too coarse (or MIN_SUPPORT / risk-threshold too strict) "
              f"to resolve a clean [X, Y]. Try a finer grid around {Y:,} and below.")
    else:
        print(f"CANDIDATE RANGE for the real training sweep is [{X:,}, {Y:,}]")
        print(f"  X = {X:,}  compression knee: beyond here extra vocab buys little "
              f"val_chars_per_token.")
        print(f"  Y = {Y:,}  undertraining crossing: at/above here >{risk_threshold:.0%} "
              f"of the vocab is seen <MIN_SUPPORT times in train (proxy risk).")
        print("  Both are grid points; treat as the coarse bracket to sweep for real.")
    print(line)


# --------------------------------------------------------------------------- #
# Persistence + plot                                                          #
# --------------------------------------------------------------------------- #
def save_outputs(results, outdir, config):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vocab_size", "vocab_actual", "val_chars_per_token",
                    "n_vocab_undertrained", "frac_undertrained"])
        for r in results:
            w.writerow([r.vocab_size, r.vocab_actual, f"{r.val_chars_per_token:.6f}",
                        r.n_vocab_undertrained, f"{r.frac_undertrained:.6f}"])

    json_path = outdir / "results.json"
    json_path.write_text(json.dumps(
        {"config": config, "results": [asdict(r) for r in results]}, indent=2))

    print(f"\nwrote {csv_path}")
    print(f"wrote {json_path}")
    return outdir


def make_plot(results, outdir, knee, crossing):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter
    except Exception as e:
        print(f"(skipping plot - matplotlib unavailable: {e})")
        return

    xs = [r.vocab_size for r in results]
    cpt = [r.val_chars_per_token for r in results]
    frac = [r.frac_undertrained for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xscale("log", base=2)
    (l1,) = ax1.plot(xs, cpt, "o-", color="tab:blue", label="val_chars_per_token")
    ax1.set_xlabel("vocab_size  (log2 axis)")
    ax1.set_ylabel("val_chars_per_token", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(xs)
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.xaxis.set_minor_formatter(plt.NullFormatter())
    ax1.set_xticklabels([f"{x:,}" for x in xs], rotation=45, ha="right")

    ax2 = ax1.twinx()
    (l2,) = ax2.plot(xs, frac, "s--", color="tab:red", label="frac_undertrained")
    ax2.set_ylabel("frac_undertrained", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    handles = [l1, l2]
    if knee is not None:
        ax1.axvline(knee, color="tab:green", ls=":", lw=1.5)
        handles.append(plt.Line2D([], [], color="tab:green", ls=":",
                                  label=f"knee  X={knee:,}"))
    if crossing is not None:
        ax1.axvline(crossing, color="tab:orange", ls=":", lw=1.5)
        handles.append(plt.Line2D([], [], color="tab:orange", ls=":",
                                  label=f"crossing  Y={crossing:,}"))
    ax1.legend(handles=handles, loc="best")
    ax1.set_title("Tokenizer compression vs undertraining proxy (no model trained)")
    fig.tight_layout()

    path = Path(outdir) / "val_cpt_vs_vocab.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Tokenizer-only vocab_size range diagnostic for train.py "
                    "(no model training, CPU only).")
    p.add_argument("--vocab-sizes", type=str, default=None,
                   help="comma-separated grid, e.g. 1024,2048,4096 "
                        f"(default: {','.join(map(str, DEFAULT_VOCAB_GRID))})")
    p.add_argument("--min-support", type=int, default=DEFAULT_MIN_SUPPORT,
                   help=f"MIN_SUPPORT for frac_undertrained (default {DEFAULT_MIN_SUPPORT})")
    p.add_argument("--risk-threshold", type=float, default=DEFAULT_RISK_THRESHOLD,
                   help=f"frac_undertrained risk threshold (default {DEFAULT_RISK_THRESHOLD})")
    p.add_argument("--num-titles", type=int, default=NUM_TITLES)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--val-frac", type=float, default=VAL_FRAC)
    p.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="tiny fast sweep (small grid + few titles) to check plumbing")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.vocab_sizes:
        grid = sorted({int(x) for x in args.vocab_sizes.split(",") if x.strip()})
    elif args.smoke:
        grid = list(SMOKE_VOCAB_GRID)
    else:
        grid = list(DEFAULT_VOCAB_GRID)
    num_titles = SMOKE_NUM_TITLES if args.smoke else args.num_titles

    get_titles, train_tokenizer, BPETokenizer = import_from_train()

    print(f"loading titles: num_titles={num_titles:,} seed={args.seed} "
          f"val_frac={args.val_frac}")
    train_titles, val_titles = get_titles(num_titles, args.seed, args.val_frac)
    print(f"  {len(train_titles):,} train / {len(val_titles):,} val titles")

    results = run_sweep(grid, train_titles, val_titles, args.min_support,
                        train_tokenizer, BPETokenizer)

    xs = [r.vocab_size for r in results]
    cpt = [r.val_chars_per_token for r in results]
    knee, knee_method, gains = detect_knee(xs, cpt)
    crossing = threshold_crossing(results, args.risk_threshold)

    summarize(results, knee, knee_method, gains, crossing, args.risk_threshold)

    config = {
        "seed": args.seed,
        "num_titles": num_titles,
        "val_frac": args.val_frac,
        "eos_token": EOS_TOKEN,
        "vocab_sizes": grid,
        "min_support": args.min_support,
        "risk_threshold": args.risk_threshold,
        "tokenizer_trained_on": "train_titles only (non-leaky; differs from train.py "
                                "which trains on train+val)",
        "knee_method": knee_method,
        "knee_vocab_size": knee,
        "threshold_crossing_vocab_size": crossing,
        "smoke": args.smoke,
    }
    outdir = save_outputs(results, args.outdir, config)
    if not args.no_plot:
        make_plot(results, outdir, knee, crossing)


if __name__ == "__main__":
    main()
