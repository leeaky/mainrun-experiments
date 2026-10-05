"""
Plain-assert unit tests for config.py (override loader + key validation).
No tokenizer, no torch, no dataset. Run:

    python3 test_config.py
"""
import json
import tempfile
from pathlib import Path

import config as cfg


def _write(obj) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(obj, f)
    f.close()
    return f.name


def test_none_path_returns_exact_defaults():
    assert cfg.load_config(None) == cfg.Hyperparameters()


def test_empty_object_returns_exact_defaults():
    assert cfg.load_config(_write({})) == cfg.Hyperparameters()


def test_override_applies_single_field():
    hp = cfg.load_config(_write({"lr": 4e-4}))
    assert hp.lr == 4e-4
    assert hp.batch_size == cfg.Hyperparameters().batch_size  # others untouched


def test_run_label_is_overridable():
    assert cfg.load_config(_write({"run_label": "adamw-test"})).run_label == "adamw-test"


def test_unknown_key_raises_and_names_the_key():
    try:
        cfg.load_config(_write({"lr": 4e-4, "bogus_knob": 1}))
    except cfg.ConfigError as e:
        assert "bogus_knob" in str(e)
    else:
        raise AssertionError("expected ConfigError for unknown key")


def test_each_protected_key_raises():
    for key, val in [
        ("epochs", 3), ("seed", 42), ("num_titles", 1000),
        ("val_frac", 0.2), ("log_file", "/tmp/x.log"),
    ]:
        try:
            cfg.load_config(_write({key: val}))
        except cfg.ConfigError as e:
            assert key in str(e), (key, str(e))
        else:
            raise AssertionError(f"expected ConfigError for protected key {key}")


def test_non_object_json_raises():
    try:
        cfg.load_config(_write([1, 2, 3]))
    except cfg.ConfigError:
        pass
    else:
        raise AssertionError("expected ConfigError for non-object JSON")


def test_wrong_type_value_raises():
    try:
        cfg.load_config(_write({"batch_size": "sixtyfour"}))
    except cfg.ConfigError:
        pass
    else:
        raise AssertionError("expected ConfigError for wrong-typed value")


def test_protected_check_precedes_type_check():
    # a protected key with a valid-typed value must still be rejected as protected
    try:
        cfg.load_config(_write({"epochs": 7}))
    except cfg.ConfigError as e:
        assert "epochs" in str(e)
    else:
        raise AssertionError("expected ConfigError")


def test_adamw_fields_have_baseline_safe_defaults():
    hp = cfg.Hyperparameters()
    assert hp.optimizer == "sgd"
    assert (hp.beta1, hp.beta2) == (0.9, 0.95)
    assert hp.warmup_frac == 0.06
    assert hp.lr_floor_frac == 0.1


def test_optimizer_accepts_adamw():
    assert cfg.load_config(_write({"optimizer": "adamw"})).optimizer == "adamw"


def test_optimizer_rejects_unknown_choice():
    try:
        cfg.load_config(_write({"optimizer": "lion"}))
    except cfg.ConfigError as e:
        assert "optimizer" in str(e) and "lion" in str(e)
    else:
        raise AssertionError("expected ConfigError for optimizer='lion'")


def test_betas_are_plain_floats_from_json():
    hp = cfg.load_config(_write({"beta1": 0.9, "beta2": 0.99}))
    assert hp.beta2 == 0.99


def test_optimizer_accepts_muon():
    assert cfg.load_config(_write({"optimizer": "muon"})).optimizer == "muon"


def test_muon_and_ema_field_defaults():
    hp = cfg.Hyperparameters()
    assert hp.muon_lr == 0.02
    assert hp.muon_momentum == 0.95
    assert hp.ema_decay == 0.0


def test_ema_decay_rejects_non_float():
    try:
        cfg.load_config(_write({"ema_decay": "high"}))
    except cfg.ConfigError:
        pass
    else:
        raise AssertionError("expected ConfigError")


def test_muon_lr_overridable():
    hp = cfg.load_config(_write({"optimizer": "muon", "muon_lr": 0.05}))
    assert hp.muon_lr == 0.05


def test_session_b_field_defaults():
    hp = cfg.Hyperparameters()
    assert hp.muon_ns_steps == 5
    assert hp.weight_decay_embeddings is True
    assert hp.mlp_kind == "gelu"
    assert hp.qk_norm is False
    assert hp.wsd_decay_shape == "linear"


def test_mlp_kind_and_wsd_shape_choices():
    assert cfg.load_config(_write({"mlp_kind": "swiglu"})).mlp_kind == "swiglu"
    assert cfg.load_config(_write({"wsd_decay_shape": "invsqrt"})).wsd_decay_shape == "invsqrt"
    for bad in ({"mlp_kind": "relu"}, {"wsd_decay_shape": "cosine"}):
        try:
            cfg.load_config(_write(bad))
        except cfg.ConfigError:
            pass
        else:
            raise AssertionError(f"expected ConfigError for {bad}")


def test_weight_decay_embeddings_rejects_non_bool():
    try:
        cfg.load_config(_write({"weight_decay_embeddings": 0}))
    except cfg.ConfigError:
        pass
    else:
        raise AssertionError("expected ConfigError")


def test_wsd_factor_invsqrt_drops_sooner():
    # warmup=60, post=940, stable=564 -> decay runs steps 624..1000
    lin = cfg.wsd_factor(880, 1000, 0.06, 0.1, 0.6)
    isq = cfg.wsd_factor(880, 1000, 0.06, 0.1, 0.6, shape="invsqrt")
    assert isq < lin                                   # 1-sqrt is below linear mid-decay
    assert abs(cfg.wsd_factor(603, 1000, 0.06, 0.1, 0.6, shape="invsqrt") - 1.0) < 1e-9
    end = cfg.wsd_factor(999, 1000, 0.06, 0.1, 0.6, shape="invsqrt")
    assert 0.1 <= end < 0.2                             # decays to ~floor


def test_warmup_cosine_factor_ramps_then_decays():
    f = cfg.warmup_cosine_factor
    # warmup = round(0.06 * 1000) = 60
    assert abs(f(0, 1000, 0.06, 0.1) - 1 / 60) < 1e-9      # first step, tiny
    assert abs(f(59, 1000, 0.06, 0.1) - 1.0) < 1e-9        # end of warmup, peak
    assert abs(f(60, 1000, 0.06, 0.1) - 1.0) < 1e-9        # cosine starts at peak
    end = f(999, 1000, 0.06, 0.1)
    assert 0.1 <= end < 0.11                                # ends at/just above floor
    mid = f(500, 1000, 0.06, 0.1)
    assert 0.1 < mid < 1.0                                  # monotone-ish descent
    assert f(100, 1000, 0.06, 0.1) > mid                    # earlier post-warmup is higher


def test_warmup_cosine_factor_zero_warmup():
    # warmup_frac 0 -> no ramp, factor starts at the cosine peak
    assert abs(cfg.warmup_cosine_factor(0, 1000, 0.0, 0.1) - 1.0) < 1e-9


def test_scaled_residual_init_default_false_and_toggleable():
    assert cfg.Hyperparameters().scaled_residual_init is False
    assert cfg.load_config(_write({"scaled_residual_init": True})).scaled_residual_init is True


def test_scaled_residual_init_rejects_non_bool():
    try:
        cfg.load_config(_write({"scaled_residual_init": 1}))  # int, not bool
    except cfg.ConfigError:
        pass
    else:
        raise AssertionError("expected ConfigError for int where bool expected")


def test_tokenizer_scope_default_and_choices():
    assert cfg.Hyperparameters().tokenizer_scope == "train_and_val"
    assert cfg.load_config(_write({"tokenizer_scope": "train_only"})).tokenizer_scope == "train_only"


def test_tokenizer_scope_rejects_unknown():
    try:
        cfg.load_config(_write({"tokenizer_scope": "val_only"}))
    except cfg.ConfigError as e:
        assert "tokenizer_scope" in str(e)
    else:
        raise AssertionError("expected ConfigError")


def test_shuffle_batches_default_false_and_toggle():
    assert cfg.Hyperparameters().shuffle_batches is False
    assert cfg.load_config(_write({"shuffle_batches": True})).shuffle_batches is True


def test_shuffle_batches_rejects_non_bool():
    try:
        cfg.load_config(_write({"shuffle_batches": "yes"}))
    except cfg.ConfigError:
        pass
    else:
        raise AssertionError("expected ConfigError")


def test_deprioritised_toggle_defaults():
    hp = cfg.Hyperparameters()
    assert hp.attn_impl == "manual"
    assert hp.grad_clip_norm == 1.0
    assert hp.use_amp is False
    assert hp.lr_schedule == "cosine"
    assert hp.stable_frac == 0.6


def test_attn_impl_and_lr_schedule_choices():
    assert cfg.load_config(_write({"attn_impl": "sdpa"})).attn_impl == "sdpa"
    assert cfg.load_config(_write({"lr_schedule": "wsd"})).lr_schedule == "wsd"
    for bad in ({"attn_impl": "flash"}, {"lr_schedule": "linear"}):
        try:
            cfg.load_config(_write(bad))
        except cfg.ConfigError:
            pass
        else:
            raise AssertionError(f"expected ConfigError for {bad}")


def test_wsd_factor_shape():
    f = cfg.wsd_factor
    # warmup=60, post=940, stable=round(0.6*940)=564
    assert abs(f(0, 1000, 0.06, 0.1, 0.6) - 1 / 60) < 1e-9      # warmup ramp
    assert abs(f(60, 1000, 0.06, 0.1, 0.6) - 1.0) < 1e-9        # start of stable
    assert abs(f(600, 1000, 0.06, 0.1, 0.6) - 1.0) < 1e-9       # mid stable
    end = f(999, 1000, 0.06, 0.1, 0.6)
    assert 0.1 <= end < 0.11                                     # decayed to floor


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
