"""PCA decomposition (α₁ tool) — numpy-only via ``numpy.linalg.svd``.

Multivariate decomposition of several sector time series into principal
components — what the β₁ planner never had (moving averages only), what we
(α₁) use to see cross-sector structure (PLAN-23 §一 kernel/ vs extras/).
"""

import numpy as np

from ..kernel._types import SpectrumResult, Verdict

_MIN_OBS = 8


def pca_decompose(data, n_components: int | None = None) -> SpectrumResult:
    """PCA over multiple time series.

    ``data``: dict ``{name: series}`` (aligned to the shortest series) or a
    2-D ndarray with variables as rows and time as columns.
    """
    if isinstance(data, dict):
        if not data:
            return SpectrumResult(values={"components": []}, verdict=Verdict.UNKNOWN,
                                  confidence_reason="empty input")
        min_len = min(len(v) for v in data.values())
        names = list(data.keys())
        X = np.array([np.asarray(data[k], dtype=np.float64)[:min_len] for k in names])
    else:
        X = np.asarray(data, dtype=np.float64)
        if X.ndim != 2:
            return SpectrumResult(values={"components": []}, verdict=Verdict.UNKNOWN,
                                  confidence_reason="input must be 2-D")
        names = [f"var_{i}" for i in range(X.shape[0])]

    n_vars, n_obs = X.shape
    if n_obs < _MIN_OBS or n_vars < 2:
        return SpectrumResult(values={"components": []}, verdict=Verdict.UNKNOWN,
                              confidence_reason=f"need ≥2 variables and ≥{_MIN_OBS} obs, got {n_vars}×{n_obs}")

    Xc = X - X.mean(axis=1, keepdims=True)
    # SVD on centered matrix (variables × observations)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2
    total = var.sum()
    ratio = var / total if total > 0 else np.zeros_like(var)

    k = n_components or min(3, len(S))
    components = []
    for i in range(k):
        loadings = U[:, i]
        top = np.argsort(np.abs(loadings))[::-1][:3]
        components.append({
            "index": i,
            "explained_variance_ratio": float(ratio[i]),
            "loadings": loadings.tolist(),
            "top_variables": [{"name": names[j], "loading": float(loadings[j])} for j in top],
        })

    first = float(ratio[0]) if len(ratio) else 0.0
    if first >= 0.5:
        verdict = Verdict.ASSERTED
        reason = f"PC1 explains {first:.2f} of variance ({n_vars} vars × {n_obs} obs)"
    elif first >= 0.2:
        verdict = Verdict.CONTESTED
        reason = f"PC1 explains only {first:.2f} — structure diffuse"
    else:
        verdict = Verdict.UNKNOWN
        reason = f"PC1 explains {first:.2f} — no dominant structure"

    return SpectrumResult(
        values={
            "components": components,
            "explained_variance_ratio": ratio.tolist(),
            "variable_names": names,
        },
        verdict=verdict,
        confidence_reason=reason,
    )
