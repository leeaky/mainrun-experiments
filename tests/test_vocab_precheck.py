"""
Plain-assert unit tests for the analysis helpers in vocab_precheck.py.

These cover the pure logic (knee detection, threshold crossing, and the
range-ordering decisions in summarize()) with synthetic arrays - no tokenizer,
no dataset, no network. Run:

    python3 test_vocab_precheck.py

The tokenizer/data path (measure_one / run_sweep) is not unit-tested here; it is
verified by `python3 vocab_precheck.py --smoke`.
"""

import io
import contextlib

import vocab_precheck as vp


def test_manual_knee_on_diminishing_returns():
    xs = [1024, 2048, 4096, 8192, 16384]
    ys = [3.0, 3.8, 4.2, 4.3, 4.35]          # big early gains, then flat
    knee, method, gains = vp.detect_knee(xs, ys)
    assert len(gains) == 4
    assert gains[0][0] == 2048 and abs(gains[0][1] - 0.8) < 1e-9
    # kneed may or may not be installed; either way a knee should land in the elbow.
    assert knee in (2048, 4096), (knee, method)


def test_manual_knee_when_kneed_absent_is_deterministic():
    # Force the manual path and check the "first step below 20% of peak gain" rule.
    xs = [256, 512, 1024, 2048]
    ys = [1.0, 2.0, 2.05, 2.07]               # peak gain 1.0 at ->512; ->1024 is 0.05
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def no_kneed(name, *a, **k):
        if name == "kneed":
            raise ImportError("blocked for test")
        return real_import(name, *a, **k)

    import builtins
    orig = builtins.__import__
    builtins.__import__ = no_kneed
    try:
        knee, method, _ = vp.detect_knee(xs, ys)
    finally:
        builtins.__import__ = orig
    assert method.startswith("manual per-doubling gain")
    assert knee == 512, knee                  # point before the first flat step


def test_threshold_crossing_basic_and_never():
    R = vp.SweepResult
    results = [
        R(1024, 1024, 4.0, 0, 0.05),
        R(2048, 2048, 4.2, 0, 0.20),
        R(4096, 4096, 4.3, 0, 0.40),
    ]
    assert vp.threshold_crossing(results, 0.25) == 4096
    assert vp.threshold_crossing(results, 0.99) is None
    # strictly greater-than, not >=
    assert vp.threshold_crossing([R(1024, 1024, 4.0, 0, 0.25)], 0.25) is None


def _summary_text(results, knee, crossing, thr=0.25):
    buf = io.StringIO()
    gains = [(results[i + 1].vocab_size,
              results[i + 1].val_chars_per_token - results[i].val_chars_per_token)
             for i in range(len(results) - 1)]
    with contextlib.redirect_stdout(buf):
        vp.summarize(results, knee, "test", gains, crossing, thr)
    return buf.getvalue()


def test_summary_ordered_range():
    R = vp.SweepResult
    results = [R(v, v, 3.0 + i * 0.3, 0, i * 0.1) for i, v in enumerate([1024, 2048, 4096, 8192])]
    txt = _summary_text(results, knee=2048, crossing=8192)
    assert "candidate range for the real training sweep is [2,048, 8,192]" in txt.lower()


def test_summary_flags_bad_ordering():
    R = vp.SweepResult
    results = [R(v, v, 3.0, 0, 0.0) for v in (1024, 2048, 4096)]
    txt = _summary_text(results, knee=4096, crossing=2048)
    assert "no sensible ordered range" in txt.lower()


def test_summary_flags_unresolved():
    R = vp.SweepResult
    results = [R(v, v, 3.0, 0, 0.0) for v in (1024, 2048, 4096)]
    txt = _summary_text(results, knee=2048, crossing=None)
    assert "not fully resolved" in txt.lower()
    assert "extend the grid upward" in txt.lower()


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
