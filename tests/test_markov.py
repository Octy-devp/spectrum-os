import numpy as np
import pytest

from spectrum_os.quantum.markov import (
    count_transitions,
    count_transitions_soft,
    estimate_rate_matrix,
    sample_rate_matrix,
    hitting_times,
    absorption_probabilities,
    smoothed_role_posteriors,
    stationary_distribution,
    entropy_rate,
    spectral_gap,
    summarize_samples,
)
from spectrum_os.quantum.multigraph import ROLES
from spectrum_os.quantum.dca_grammar import DCA_GRAMMAR


def test_count_transitions_basic():
    seqs = [
        ["crisis", "lag", "alternative", "direction", "crisis"],
        ["crisis", "crisis", "lag"],
    ]
    counts, anomalies = count_transitions(seqs)
    assert isinstance(counts, np.ndarray)
    assert counts.shape == (4, 4)
    assert len(anomalies) == 0

    # Index mapping: 0: crisis, 1: lag, 2: alternative, 3: direction
    # seq 1: crisis->lag (1), lag->alternative (1), alternative->direction (1), direction->crisis (1)
    # seq 2: crisis->crisis (1), crisis->lag (1)
    # Total crisis->lag = 2, crisis->crisis = 1, etc.
    assert counts[0, 1] == 2  # crisis -> lag
    assert counts[0, 0] == 1  # crisis -> crisis
    assert counts[1, 2] == 1  # lag -> alternative
    assert counts[2, 3] == 1  # alternative -> direction
    assert counts[3, 0] == 1  # direction -> crisis


def test_count_transitions_anomalies():
    seqs = [
        ["lag", "direction"],  # forbidden: lag -> direction
        ["direction", "alternative", "crisis"],  # forbidden: direction -> alternative
    ]
    counts, anomalies = count_transitions(seqs)
    assert len(anomalies) == 2
    assert anomalies[0] == {
        "from": "lag",
        "to": "direction",
        "position": 0,
        "sequence": ["lag", "direction"],
    }
    assert anomalies[1] == {
        "from": "direction",
        "to": "alternative",
        "position": 0,
        "sequence": ["direction", "alternative", "crisis"],
    }
    # Check that forbidden transitions are NOT swallowed in counts
    assert counts[1, 3] == 1
    assert counts[3, 2] == 1


def test_estimate_rate_matrix_forbidden_zero():
    counts = np.zeros((4, 4))
    res = estimate_rate_matrix(counts, alpha=1.0, min_total_count=10.0)

    assert "matrix" in res
    assert "verdict" in res
    assert "counts" in res
    assert "total_count" in res

    P = res["matrix"]
    assert P.shape == (4, 4)
    # Verdict should be UNKNOWN since total_count is 0 < 10
    assert res["verdict"] == "UNKNOWN"
    assert res["total_count"] == 0.0

    # Forbidden positions must be 0.0
    # lag (1) -> direction (3)
    assert P[1, 3] == 0.0
    # direction (3) -> alternative (2)
    assert P[3, 2] == 0.0

    # Check row normalization
    np.testing.assert_allclose(P.sum(axis=1), np.ones(4))


def test_estimate_rate_matrix_asserted():
    counts = np.ones((4, 4)) * 5  # total count = 80
    res = estimate_rate_matrix(counts, alpha=1.0, min_total_count=10.0)
    assert res["verdict"] == "ASSERTED"
    assert res["total_count"] == 80.0

    P = res["matrix"]
    # Row 1 (lag): allowed targets 0, 1, 2 (each has count 5, alpha 1 -> numerator 6). Sum = 18.
    # P[1, 0] = 6/18 = 1/3, P[1, 1] = 1/3, P[1, 2] = 1/3, P[1, 3] = 0
    np.testing.assert_allclose(P[1], [1 / 3, 1 / 3, 1 / 3, 0.0])

    # Row 3 (direction): allowed targets 0, 1, 3 (each count 5, alpha 1 -> 6). Sum = 18.
    # P[3, 2] = 0
    np.testing.assert_allclose(P[3], [1 / 3, 1 / 3, 0.0, 1 / 3])


def test_sample_rate_matrix():
    counts = np.ones((4, 4)) * 2
    samples = sample_rate_matrix(counts, alpha=1.0, n=50, seed=42)

    assert len(samples) == 50
    for sample in samples:
        assert sample.shape == (4, 4)
        # Forbidden cells are 0
        assert sample[1, 3] == 0.0
        assert sample[3, 2] == 0.0
        # Row sums are 1.0
        np.testing.assert_allclose(sample.sum(axis=1), np.ones(4))

    # Test reproducibility with seed
    samples_again = sample_rate_matrix(counts, alpha=1.0, n=50, seed=42)
    for s1, s2 in zip(samples, samples_again):
        np.testing.assert_array_equal(s1, s2)


def test_hitting_times_analytical():
    # 2-state analytical case
    # P = [[0.5, 0.5], [0.0, 1.0]]
    # Starting at "crisis", expected steps to reach "lag" is 1/0.5 = 2.0.
    P = np.array([[0.5, 0.5], [0.0, 1.0]])
    ht = hitting_times(P, target="lag", roles=["crisis", "lag"])
    assert ht["lag"] == 0.0
    pytest.approx(ht["crisis"], 2.0)

    # 4-state test using default ROLES
    # ROLES = ["crisis", "lag", "alternative", "direction"]
    # Chain: crisis -> lag (1.0), lag -> alternative (1.0), alternative -> direction (1.0), direction -> direction (1.0)
    P4 = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    ht4 = hitting_times(P4, target="direction")
    assert ht4["direction"] == 0.0
    np.testing.assert_allclose(ht4["alternative"], 1.0)
    np.testing.assert_allclose(ht4["lag"], 2.0)
    np.testing.assert_allclose(ht4["crisis"], 3.0)


def test_absorption_probabilities_analytical():
    # 4-state with transient ["lag", "alternative"] and absorbing ["crisis", "direction"]
    # ROLES = ["crisis", "lag", "alternative", "direction"]
    # crisis (0): absorbing (P[0,0]=1.0)
    # lag (1): P[1,0]=0.4 (crisis), P[1,2]=0.6 (alternative)
    # alternative (2): P[2,3]=1.0 (direction)
    # direction (3): absorbing (P[3,3]=1.0)
    P = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.4, 0.0, 0.6, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    ab = absorption_probabilities(P, absorbing=["crisis", "direction"])
    # Transient roles: "lag", "alternative"
    assert set(ab.keys()) == {"lag", "alternative"}
    assert set(ab["lag"].keys()) == {"crisis", "direction"}
    assert set(ab["alternative"].keys()) == {"crisis", "direction"}

    np.testing.assert_allclose(ab["lag"]["crisis"], 0.4)
    np.testing.assert_allclose(ab["lag"]["direction"], 0.6)
    np.testing.assert_allclose(ab["alternative"]["crisis"], 0.0)
    np.testing.assert_allclose(ab["alternative"]["direction"], 1.0)


def test_stationary_distribution_property():
    P = np.array([
        [0.5, 0.5],
        [0.2, 0.8],
    ])
    pi = stationary_distribution(P)
    assert pi.shape == (2,)
    np.testing.assert_allclose(np.sum(pi), 1.0)
    # pi @ P == pi
    np.testing.assert_allclose(pi @ P, pi, atol=1e-12)
    # Analytical pi = [2/7, 5/7]
    np.testing.assert_allclose(pi, [2 / 7, 5 / 7])


def test_entropy_rate_analytical():
    # Uniform 2x2: entropy rate is ln(2)
    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    rate = entropy_rate(P)
    np.testing.assert_allclose(rate, np.log(2.0))

    # Identity 2x2: deterministic transitions -> 0.0
    P_eye = np.eye(2)
    assert entropy_rate(P_eye) == 0.0


def test_spectral_gap_properties():
    # Identity matrix: all eigenvalues 1.0 -> spectral gap 0.0
    I4 = np.eye(4)
    assert spectral_gap(I4) == 0.0

    # Uniform matrix: eigenvalues [1, 0, 0, 0] -> spectral gap 1.0
    U4 = np.ones((4, 4)) * 0.25
    np.testing.assert_allclose(spectral_gap(U4), 1.0)


def test_summarize_samples():
    vals = list(np.linspace(0.0, 100.0, 101))
    summary = summarize_samples(vals)
    assert "mean" in summary
    assert "ci95_low" in summary
    assert "ci95_high" in summary

    np.testing.assert_allclose(summary["mean"], 50.0)
    np.testing.assert_allclose(summary["ci95_low"], 2.5)
    np.testing.assert_allclose(summary["ci95_high"], 97.5)


def test_demo_smoke(tmp_path):
    import os
    from scripts.markov_warfare_demo import TRACK0_PATH, run_demo

    if not os.path.exists(TRACK0_PATH):
        pytest.skip(f"Track 0 file missing: {TRACK0_PATH}")

    output_path = str(tmp_path / "markov_demo_report.json")
    report = run_demo(tree_path=TRACK0_PATH, output_path=output_path, n_samples=50, seed=42)

    assert os.path.exists(output_path)
    assert report["total_sequences"] > 0
    assert report["verdict"] in ("ASSERTED", "UNKNOWN")
    assert "entropy_rate" in report["point_estimates"]
    assert "spectral_gap" in report["point_estimates"]
    assert "entropy_rate" in report["posterior_samples"]
    assert "spectral_gap" in report["posterior_samples"]
    assert report["hitting_times_to_direction"]["direction"] == 0.0
    assert "lag" in report["absorption_probabilities"]


def test_hitting_times_disconnected_inf():
    # Disconnected state 0 (crisis) cannot reach target state 3 (direction)
    P = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ])
    ht = hitting_times(P, target="direction")
    assert ht["direction"] == 0.0
    assert ht["crisis"] == float("inf")
    assert ht["lag"] == float("inf")
    np.testing.assert_allclose(ht["alternative"], 2.0)


def test_stationary_distribution_edge_cases():
    # Zero matrix fallback to uniform distribution
    P_zero = np.zeros((4, 4))
    pi_zero = stationary_distribution(P_zero)
    np.testing.assert_allclose(pi_zero, [0.25, 0.25, 0.25, 0.25])

    # Disconnected / absorbing matrix
    P_disc = np.eye(4)
    pi_disc = stationary_distribution(P_disc)
    assert pi_disc.shape == (4,)
    np.testing.assert_allclose(np.sum(pi_disc), 1.0)


def test_smoothed_role_posteriors_analytical_2state():
    # 2-state HMM analytical test (roles 0 and 1 active)
    # P matrix: role 0->0 (0.8), 0->1 (0.2); role 1->0 (0.3), 1->1 (0.7)
    P = np.array([
        [0.8, 0.2, 0.0, 0.0],
        [0.3, 0.7, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.5, 0.5],
    ])
    emission_loglik = np.array([
        [np.log(0.9), np.log(0.1), -100.0, -100.0],
        [np.log(0.2), np.log(0.8), -100.0, -100.0],
        [np.log(0.7), np.log(0.3), -100.0, -100.0],
    ])
    res = smoothed_role_posteriors(emission_loglik, P)
    gamma = res["gamma"]
    xi = res["xi"]
    loglik = res["loglik"]

    assert gamma.shape == (3, 4)
    assert xi.shape == (2, 4, 4)
    assert np.isfinite(loglik)
    np.testing.assert_allclose(gamma.sum(axis=1), np.ones(3))
    np.testing.assert_allclose(xi.sum(axis=(1, 2)), np.ones(2))
    assert not np.isnan(gamma).any()
    assert not np.isnan(xi).any()


def test_smoothed_role_posteriors_single_obs_boundary():
    P = np.ones((4, 4)) * 0.25
    emission_loglik = np.zeros((1, 4))  # T=1
    res = smoothed_role_posteriors(emission_loglik, P)

    assert res["gamma"].shape == (1, 4)
    assert res["xi"].shape == (0, 4, 4)
    np.testing.assert_allclose(res["gamma"].sum(axis=1), [1.0])


def test_smoothed_role_posteriors_zero_prob_no_nan():
    P = np.array([
        [0.5, 0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.5, 0.5],
    ])
    # Extreme log likelihoods including -inf
    emission_loglik = np.array([
        [-np.inf, -np.inf, 0.0, 0.0],
        [-1e9, -1e9, 0.0, 0.0],
        [0.0, 0.0, -np.inf, -np.inf],
    ])
    res = smoothed_role_posteriors(emission_loglik, P)
    assert not np.isnan(res["gamma"]).any()
    assert not np.isnan(res["xi"]).any()


def test_smoothed_role_posteriors_delta_limit_consistency():
    # Delta emission (hard labels limit) -> soft count must match v1 hard count
    P = estimate_rate_matrix(np.zeros((4, 4)), alpha=1.0)["matrix"]
    seq = ["crisis", "lag", "alternative", "direction", "crisis"]
    role_to_idx = {r: i for i, r in enumerate(ROLES)}

    # Build hard delta emission loglik
    T = len(seq)
    delta_emission = np.full((T, 4), -1e9, dtype=np.float64)
    for t, r in enumerate(seq):
        delta_emission[t, role_to_idx[r]] = 0.0  # log(1.0) = 0.0

    res = smoothed_role_posteriors(delta_emission, P)
    soft_counts = count_transitions_soft(res["xi"])
    hard_counts, _ = count_transitions([seq])

    # Soft transition matrix should match hard transition matrix
    np.testing.assert_allclose(soft_counts, hard_counts, atol=1e-5)


def test_smoothed_role_posteriors_impossible_sequence_fallback():
    P = np.eye(4) * 0.25
    # T=3 with all -inf emission loglik (impossible observations)
    emission_loglik = np.full((3, 4), -np.inf, dtype=np.float64)
    res = smoothed_role_posteriors(emission_loglik, P)

    assert res["loglik"] == -np.inf
    assert res["gamma"].shape == (3, 4)
    # Each row in gamma must sum to 1.0 (uniform distribution fallback)
    np.testing.assert_allclose(res["gamma"].sum(axis=1), np.ones(3))
    np.testing.assert_allclose(res["gamma"], np.full((3, 4), 0.25))


def test_soft_counts_structural_zero_preservation():
    # Verify that structural zero transitions (lag->direction [1,3], direction->alternative [3,2])
    # remain exactly 0.0 in count_transitions_soft and subsequent estimate_rate_matrix.
    counts_init = np.zeros((4, 4))
    P = estimate_rate_matrix(counts_init, alpha=1.0)["matrix"]

    # Random emission loglik sequence
    rng = np.random.default_rng(42)
    obs_log = rng.normal(size=(10, 4))

    res = smoothed_role_posteriors(obs_log, P)
    soft_counts = count_transitions_soft(res["xi"])

    # Structural zeros must be exactly 0.0
    assert soft_counts[1, 3] == 0.0
    assert soft_counts[3, 2] == 0.0

    # Rate matrix estimated from soft_counts must also maintain 0.0
    P_new = estimate_rate_matrix(soft_counts, alpha=1.0)["matrix"]
    assert P_new[1, 3] == 0.0
    assert P_new[3, 2] == 0.0





