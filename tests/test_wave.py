"""Tests for Spectrum OS wave decomposition engine."""

import numpy as np
np.random.seed(42)

# (numpy only — no scipy, no sklearn)

from spectrum_os.kernel.wave import (
    moving_average,
    decompose,
    dominant_periods,
    correlate,
    _find_peaks,
)
from spectrum_os.kernel._types import Verdict


# ===================================================================
# moving_average
# ===================================================================

class TestMovingAverage:
    def test_odd_window(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = moving_average(data, 3)
        expected = np.array([np.nan, 2.0, 3.0, 4.0, np.nan])
        np.testing.assert_array_equal(result, expected)

    def test_even_window(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = moving_average(data, 2)
        expected = np.array([np.nan, 1.5, 2.5, 3.5, 4.5])
        np.testing.assert_array_equal(result, expected)

    def test_window_one(self):
        data = np.array([1.0, 2.0, 3.0])
        result = moving_average(data, 1)
        np.testing.assert_array_equal(result, data)

    def test_window_larger_than_data(self):
        data = np.array([1.0, 2.0])
        result = moving_average(data, 5)
        assert np.all(np.isnan(result))

    def test_empty_array(self):
        data = np.array([])
        result = moving_average(data, 3)
        assert len(result) == 0

    def test_window_zero_or_negative(self):
        data = np.array([1.0, 2.0, 3.0])
        result = moving_average(data, 0)
        assert np.all(np.isnan(result))


# ===================================================================
# _find_peaks
# ===================================================================

class TestFindPeaks:
    def test_single_peak(self):
        """Single peak at position 5 → returns [5]"""
        autocorr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 0.5, 0.4, 0.3, 0.2, 0.1])
        peaks = _find_peaks(autocorr, threshold=0.3)
        assert peaks == [5], f"Expected [5], got {peaks}"

    def test_plateau(self):
        """Flat plateau at positions 5-7 → returns [6] (midpoint)"""
        autocorr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.8, 0.8, 0.5, 0.4, 0.3])
        peaks = _find_peaks(autocorr, threshold=0.3)
        assert peaks == [6], f"Expected [6], got {peaks}"

    def test_no_peaks_below_threshold(self):
        autocorr = np.array([0.1, 0.2, 0.15, 0.2, 0.1])
        peaks = _find_peaks(autocorr, threshold=0.3)
        assert peaks == []

    def test_short_array(self):
        assert _find_peaks(np.array([0.1, 0.2])) == []


# ===================================================================
# dominant_periods
# ===================================================================

class TestDominantPeriods:
    def test_sine_wave_period_12(self):
        """12-month sine wave → includes period 12"""
        t = np.arange(120)
        signal = np.sin(2 * np.pi * t / 12)
        periods = dominant_periods(signal)
        assert 12 in periods, f"Expected period 12, got {periods}"

    def test_short_sequence(self):
        assert dominant_periods(np.array([1.0, 2.0])) == []

    def test_constant_sequence(self):
        assert dominant_periods(np.ones(50)) == []


# ===================================================================
# correlate
# ===================================================================

class TestCorrelate:
    def test_finds_lag(self):
        """Signal A leads signal B by 3 months → correlate finds lag=3"""
        # Use trend+noise (non-periodic) so the lag is unambiguous
        t = np.arange(120)
        a = 0.01 * t + np.random.randn(120) * 0.1
        # b lags a by 3 months: b[t] = a[t-3]
        b = np.zeros_like(a)
        b[3:] = a[:-3]
        result = correlate(a, b)
        assert result["lag_months"] == 3, (
            f"Expected lag=3, got {result['lag_months']} "
            f"(r={result['r']:.4f})"
        )

    def test_identical_signals(self):
        a = np.random.randn(100)
        result = correlate(a, a)
        assert result["lag_months"] == 0
        assert result["r"] > 0.99

    def test_short_arrays(self):
        result = correlate(np.array([1.0]), np.array([2.0]))
        assert result["lag_months"] == 0


# ===================================================================
# decompose
# ===================================================================

class TestDecompose:
    def test_detects_simple_period(self):
        """10 years of 12-month sine wave → dominant_periods includes 12"""
        t = np.arange(120)
        signal = np.sin(2 * np.pi * t / 12)
        result = decompose(signal)
        assert 12 in result["dominant_periods"], (
            f"Expected period 12, got {result['dominant_periods']}"
        )

    def test_verdict_asserted(self):
        """120 months of clean data → verdict ASSERTED"""
        t = np.arange(120)
        signal = np.sin(2 * np.pi * t / 12) + np.random.randn(120) * 0.1
        result = decompose(signal)
        assert result["verdict"] == Verdict.ASSERTED, (
            f"Expected ASSERTED, got {result['verdict']}: "
            f"{result['confidence_reason']}"
        )

    def test_verdict_unknown(self):
        """8 months of data → verdict UNKNOWN (len < 12)"""
        signal = np.random.randn(8)
        result = decompose(signal)
        assert result["verdict"] == Verdict.UNKNOWN, (
            f"Expected UNKNOWN, got {result['verdict']}"
        )

    def test_valid_range(self):
        t = np.arange(120)
        signal = np.sin(2 * np.pi * t / 12)
        result = decompose(signal, long_window=36, mid_window=6)
        start, end = result["valid_range"]
        assert start > 0, "Valid range should not start at 0 (NaN edges)"
        assert end < 119, "Valid range should not end at last index (NaN edges)"
        assert start < end, "Valid range should have positive length"

    def test_deviation_with_targets(self):
        t = np.arange(60)
        actual = np.sin(2 * np.pi * t / 12) + 100.0
        plan = np.full(60, 100.0)
        result = decompose(actual, targets=plan, long_window=12, mid_window=3)
        assert result["deviation"] is not None
        # deviation should be near 0 at the midpoint (plan is constant 100,
        # actual oscillates around 100)
        mid = len(actual) // 2
        mid_valid = min(mid, result["valid_range"][1])
        if mid_valid > result["valid_range"][0]:
            dev = result["deviation"][result["valid_range"][0]:mid_valid]
            assert np.all(np.isfinite(dev)), "Deviation has NaN inside valid range"

    def test_nan_in_middle_reduces_verdict(self):
        """Interior NaN should prevent ASSERTED verdict."""
        t = np.arange(60)
        signal = np.sin(2 * np.pi * t / 12)
        signal_with_nan = signal.copy()
        signal_with_nan[30] = np.nan  # interior NaN
        result = decompose(signal_with_nan, long_window=12, mid_window=3)
        # Should NOT be ASSERTED because of interior NaN
        assert result["verdict"] != Verdict.ASSERTED, (
            f"Expected not ASSERTED due to NaN, got {result['verdict']}: "
            f"{result['confidence_reason']}"
        )
