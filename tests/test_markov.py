import numpy as np
import pytest

from spectrum_os.quantum.markov import (
    count_transitions,
    estimate_rate_matrix,
    sample_rate_matrix,
    hitting_times,
    absorption_probabilities,
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



