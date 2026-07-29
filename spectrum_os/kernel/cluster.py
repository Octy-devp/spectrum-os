"""Manual k-means clustering (numpy-only) for deviation signatures.

Converts deviation signatures into fixed-length feature vectors and clusters
them using k-means++ initialisation + Lloyd iteration.
"""

import numpy as np

from ._types import Verdict

# ---------------------------------------------------------------------------
# Threshold constants (adjustable)
# ---------------------------------------------------------------------------
_MAX_INIT_ATTEMPTS = 3
_MAX_ITER = 100
_CONVERGENCE_EPS = 1e-6

# Feature vector dimension
_MAX_PERIODS = 5


def _signature_to_vector(sig: dict) -> np.ndarray:
    """Convert a deviation signature dict to a fixed-length 11-d vector.

    Components:
    * 5 values: periods, sorted, padded with zeros
    * 5 values: amplitudes, sorted by period, padded with zeros
    * 1 value: mean_deviation

    Total dimension: 5 + 5 + 1 = 11
    """
    periods = sorted(sig.get("periods", []))[:_MAX_PERIODS]
    amplitudes = sig.get("amplitudes", [])

    # Align amplitudes to sorted periods
    sorted_amps = [a for _, a in sorted(zip(sig.get("periods", []), amplitudes))]
    sorted_amps = sorted_amps[:_MAX_PERIODS]

    vec = np.zeros(_MAX_PERIODS * 2 + 1, dtype=np.float64)
    for i, p in enumerate(periods):
        vec[i] = float(p)
    for i, a in enumerate(sorted_amps):
        vec[_MAX_PERIODS + i] = float(a)
    vec[-1] = float(sig.get("mean_deviation", 0.0))

    return vec


def _kmeans_plus_plus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Simplified k-means++ centroid initialisation."""
    n = X.shape[0]
    if k >= n:
        # Fallback: pick all points as centroids
        return X.copy()

    # Pick first centroid uniformly at random
    centroids = [X[rng.integers(0, n)]]

    for _ in range(1, k):
        # Compute squared distances to nearest centroid
        dists = np.full(n, np.inf)
        for c in centroids:
            d = np.sum((X - c) ** 2, axis=1)
            dists = np.minimum(dists, d)

        # Probability proportional to distance squared
        total = np.sum(dists)
        if total == 0:
            # All points at same location — pick randomly
            centroids.append(X[rng.integers(0, n)])
        else:
            probs = dists / total
            idx = rng.choice(n, p=probs)
            centroids.append(X[idx])

    return np.array(centroids)


def _lloyd_iteration(X: np.ndarray, centroids: np.ndarray,
                     max_iter: int, eps: float) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Run Lloyd's algorithm until convergence.

    Returns
    -------
    (labels, final_centroids, inertia, n_iter)
    """
    n = X.shape[0]
    k = centroids.shape[0]
    labels = np.zeros(n, dtype=np.intp)
    prev_centroids = centroids.copy()

    for iteration in range(max_iter):
        # Assignment step
        for i in range(n):
            dists = np.sum((X[i] - centroids) ** 2, axis=1)
            labels[i] = int(np.argmin(dists))

        # Update step
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centroids[j] = np.mean(X[mask], axis=0)

        # Convergence check
        shift = np.sum((centroids - prev_centroids) ** 2)
        prev_centroids = centroids.copy()

        if shift < eps:
            return labels, centroids, _inertia(X, labels, centroids), iteration + 1

    return labels, centroids, _inertia(X, labels, centroids), max_iter


def _inertia(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """Sum of squared distances from points to their assigned centroids."""
    total = 0.0
    for i in range(X.shape[0]):
        total += float(np.sum((X[i] - centroids[labels[i]]) ** 2))
    return total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(signatures: list[dict], n_clusters: int = 5,
        method: str = "kmeans") -> dict:
    """Cluster deviation signatures.

    Parameters
    ----------
    signatures
        List of dicts as returned by :func:`template._signature`.  Each must
        contain ``periods``, ``amplitudes``, ``mean_deviation``.  A signature
        derived from a synthetic sector should carry ``synthetic: True`` —
        if any input does, the result is stamped ``synthetic_input: True``
        (PLAN-23 §7.3 guardrail).
    n_clusters
        Number of clusters (``k``).  Clamped to ``[1, n_samples]``.
    method
        Only ``"kmeans"`` is supported.

    Returns
    -------
    dict with keys ``labels``, ``centroids``, ``inertia``, ``n_iter``,
    ``stability``, ``synthetic_input``.
    """
    synthetic_input = any(bool(sig.get("synthetic")) for sig in signatures)

    if not signatures:
        return {
            "labels": {},
            "centroids": [],
            "inertia": 0.0,
            "n_iter": 0,
            "stability": None,
            "synthetic_input": False,
            "verdict": Verdict.UNKNOWN.value,
        }

    n = len(signatures)
    k = max(1, min(n_clusters, n))

    # Build feature matrix
    X = np.zeros((n, _MAX_PERIODS * 2 + 1), dtype=np.float64)
    for i, sig in enumerate(signatures):
        X[i] = _signature_to_vector(sig)

    rng = np.random.default_rng(42)

    best_inertia = float('inf')
    best_labels = None
    best_centroids = None
    best_n_iter = 0

    all_label_sets: list[np.ndarray] = []

    for _ in range(_MAX_INIT_ATTEMPTS):
        centroids = _kmeans_plus_plus(X, k, rng)
        labels, final_centroids, inert, n_iter = _lloyd_iteration(
            X, centroids, _MAX_ITER, _CONVERGENCE_EPS
        )

        all_label_sets.append(labels.copy())

        if inert < best_inertia:
            best_inertia = inert
            best_labels = labels
            best_centroids = final_centroids
            best_n_iter = n_iter

    # Stability: fraction of runs with same labels as best run
    if len(all_label_sets) > 1:
        matches = []
        for ls in all_label_sets:
            matches.append(float(np.mean(ls == best_labels)))
        stability = float(np.mean(matches))
    else:
        stability = None

    # Build centroid dicts (convert back to signature-like form)
    centroid_list: list[dict] = []
    for c in best_centroids:
        centroid_list.append({
            "periods": [int(round(c[i])) for i in range(_MAX_PERIODS) if c[i] != 0],
            "amplitudes": [float(c[_MAX_PERIODS + i]) for i in range(_MAX_PERIODS)
                           if c[_MAX_PERIODS + i] != 0],
            "mean_deviation": float(c[-1]),
        })

    # Build labels dict (index -> cluster)
    labels_dict: dict[int, int] = {}
    for i, lbl in enumerate(best_labels):
        labels_dict[i] = int(lbl)

    # Verdict: asserted if stability is high (>=0.8), contested if moderate,
    # unknown if unstable or single-run (stability is None).
    if stability is None:
        verdict = Verdict.UNKNOWN
    elif stability >= 0.8:
        verdict = Verdict.ASSERTED
    elif stability >= 0.5:
        verdict = Verdict.CONTESTED
    else:
        verdict = Verdict.UNKNOWN

    return {
        "labels": labels_dict,
        "centroids": centroid_list,
        "inertia": best_inertia,
        "n_iter": best_n_iter,
        "stability": stability,
        "synthetic_input": synthetic_input,
        "verdict": verdict.value,
    }
