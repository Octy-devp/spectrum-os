"""FFT spectrum (α₁ tool) — numpy-only, no scipy.

β₁'s kernel decomposes with moving averages; Fourier decomposition is an
α₁ instrument. Use this to *analyze* β₁ data, not to simulate how β₁ planners
saw it (PLAN-23 §一 kernel/ vs extras/).
"""

import numpy as np

from ..kernel._types import SpectrumResult, Verdict

_MIN_POINTS = 8
_ASSERT_N = 24
_ASSERT_PEAK_SHARE = 0.2


def fft_decompose(timeseries, top_k: int = 5) -> SpectrumResult:
    """Compute the FFT amplitude/phase spectrum and top-k peaks.

    Returns a :class:`SpectrumResult` whose ``values`` contains ``peaks``
    (each with ``freq``, ``period``, ``amplitude``, ``amplitude_share``,
    ``phase_angle``) plus the full ``magnitude``/``phase`` spectra.
    """
    x = np.asarray(timeseries, dtype=np.float64)
    n = x.size

    if n < _MIN_POINTS or np.all(x == x[0]):
        return SpectrumResult(
            values={"peaks": []},
            verdict=Verdict.UNKNOWN,
            confidence_reason="too few points or constant series",
        )

    x = x - x.mean()
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)

    # skip DC bin
    mags = magnitude[1:]
    total = mags.sum()
    top_idx = np.argsort(mags)[::-1][:top_k]
    peaks = [
        {
            "freq": float(freqs[i + 1]),
            "period": float(1.0 / freqs[i + 1]) if freqs[i + 1] > 0 else float("inf"),
            "amplitude": float(mags[i]),
            "amplitude_share": float(mags[i] / total) if total > 0 else 0.0,
            "phase_angle": float(phase[i + 1]),
        }
        for i in top_idx
    ]

    top_share = peaks[0]["amplitude_share"] if peaks else 0.0
    if n >= _ASSERT_N and top_share >= _ASSERT_PEAK_SHARE:
        verdict = Verdict.ASSERTED
        reason = f"n={n}, top peak share {top_share:.2f}"
    elif n >= _MIN_POINTS * 2:
        verdict = Verdict.CONTESTED
        reason = f"n={n}, pattern weak (top share {top_share:.2f})"
    else:
        verdict = Verdict.UNKNOWN
        reason = f"n={n} too short for spectral claim"

    return SpectrumResult(
        values={
            "peaks": peaks,
            "magnitude": magnitude.tolist(),
            "phase": phase.tolist(),
            "freqs": freqs.tolist(),
        },
        verdict=verdict,
        confidence_reason=reason,
        dominant_periods=[round(p["period"]) for p in peaks if np.isfinite(p["period"])],
    )
