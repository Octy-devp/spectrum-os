"""Tests for kernel wave.extrapolate (PLAN-23 無條件預報 op)."""

import numpy as np
import pytest

from spectrum_os.kernel import osc, wave
from spectrum_os.kernel._types import Verdict


def _sine(n=120, period=12.0, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    return np.sin(2 * np.pi * np.arange(n) / period) + noise * rng.normal(size=n)


# ---------------------------------------------------------------------------
# core forecast accuracy
# ---------------------------------------------------------------------------

class TestForecastAccuracy:
    def test_sine_forecast_tracks_truth(self):
        x = _sine()
        res = wave.extrapolate(x, horizon=12, seed=42)
        assert res["verdict"] == Verdict.ASSERTED
        truth = np.sin(2 * np.pi * np.arange(120, 132) / 12.0)
        err = np.abs(res["forecast"] - truth)
        assert err.mean() < 0.2
        assert err.max() < 0.5

    def test_periods_detected(self):
        res = wave.extrapolate(_sine(), horizon=12, seed=42)
        assert 12 in res["dominant_periods"]

    def test_confidence_band_covers_truth(self):
        x = _sine()
        res = wave.extrapolate(x, horizon=12, seed=42)
        truth = np.sin(2 * np.pi * np.arange(120, 132) / 12.0)
        inside = np.sum((truth >= res["lower"]) & (truth <= res["upper"]))
        assert inside >= 10  # ≥83% of truth points within the 80% band

    def test_linear_trend_slope(self):
        # 長序列減少邊緣效應；trend_slope 擬合在移動平均長波上，
        # 長波邊緣以最近值填充會壓平首尾 → 斜率被系統性低估（已知的 smoothing bias）
        x = np.arange(200, dtype=np.float64) * 0.5 + 10
        res = wave.extrapolate(x, horizon=6, seed=1)
        assert 0.3 < res["trend_slope"] <= 0.5  # 方向正確、幅度被長波邊緣壓低
        assert res["forecast"][-1] > res["forecast"][0]  # 預測延續上升趨勢


# ---------------------------------------------------------------------------
# degenerate inputs
# ---------------------------------------------------------------------------

class TestDegenerate:
    def test_short_series_unknown_with_nan(self):
        res = wave.extrapolate(np.array([1.0, 2.0, 3.0]), horizon=12)
        assert res["verdict"] == Verdict.UNKNOWN
        assert np.all(np.isnan(res["forecast"]))

    def test_constant_series_not_asserted(self):
        res = wave.extrapolate(np.full(80, 7.0), horizon=6, seed=0)
        assert res["verdict"] != Verdict.ASSERTED

    def test_noise_only_series_not_asserted(self):
        rng = np.random.default_rng(3)
        res = wave.extrapolate(rng.normal(size=100), horizon=6, seed=0)
        assert res["verdict"] != Verdict.ASSERTED


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_result(self):
        x = _sine()
        r1 = wave.extrapolate(x, horizon=12, seed=42)
        r2 = wave.extrapolate(x, horizon=12, seed=42)
        np.testing.assert_array_equal(r1["forecast"], r2["forecast"])
        np.testing.assert_array_equal(r1["lower"], r2["lower"])


# ---------------------------------------------------------------------------
# synthetic ceiling (wave_ops wrapper)
# ---------------------------------------------------------------------------

class TestSyntheticCeiling:
    def _register(self, name, synthetic):
        meta = None
        if synthetic:
            meta = {"synthetic": True, "anchors": {}, "generated_by": "test", "seed": 42}
        return osc.sectors.create(name, _sine().tolist(), meta=meta)

    def test_synthetic_capped_to_contested(self):
        sec = self._register("xt_synth_cap_test", synthetic=True)
        res = osc.wave.extrapolate(sec.id, horizon=12, seed=42)
        assert res["synthetic_input"] is True
        assert res["verdict"] == Verdict.CONTESTED  # R² 高本是 ASSERTED → 被封頂

    def test_real_sector_uncapped(self):
        sec = self._register("xt_real_uncap_test", synthetic=False)
        res = osc.wave.extrapolate(sec.id, horizon=12, seed=42)
        assert res["synthetic_input"] is False
        assert res["verdict"] == Verdict.ASSERTED

    def test_synthetic_meta_incomplete_refused(self):
        with pytest.raises(ValueError, match="missing required keys"):
            osc.sectors.create("xt_bad_meta_test", _sine().tolist(),
                               meta={"synthetic": True})
