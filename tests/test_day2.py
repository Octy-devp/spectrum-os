"""Tests for Day 2 modules: wave_ops, template, branch, cluster, verify."""

import numpy as np
np.random.seed(42)

import pytest

# ---------------------------------------------------------------------------
# Fixtures — register test sectors
# ---------------------------------------------------------------------------

from spectrum_os.kernel import sector, wave, wave_ops, template, branch, cluster, verify
from spectrum_os.kernel._types import Verdict


@pytest.fixture(autouse=True)
def reset_sectors():
    """Clear sector store before each test."""
    # Access the internal _sectors dict via the module
    sector._sectors.clear()
    yield


@pytest.fixture(autouse=True)
def reset_templates():
    """Clear template library before each test."""
    template._templates.clear()
    yield


@pytest.fixture(autouse=True)
def reset_state_log():
    """Clear verify state log before each test."""
    verify._state_log.clear()
    yield


# ===================================================================
# wave_ops
# ===================================================================

class TestWaveOps:
    def _make_sector(self, n_months: int = 120, period: int = 12):
        t = np.arange(n_months)
        signal = np.sin(2 * np.pi * t / period) + 100.0
        plan = np.full(n_months, 100.0)
        sector.create("test_sector", signal.tolist(), plan.tolist())

    def test_decompose_by_sector_id(self):
        self._make_sector()
        result = wave_ops.decompose("test_sector")
        assert isinstance(result.values["longwave"], np.ndarray)
        assert isinstance(result.values["midwave"], np.ndarray)
        assert isinstance(result.values["shortwave"], np.ndarray)
        assert result.values["deviation"] is not None
        assert 12 in result.dominant_periods

    def test_correlate_by_sector_id(self):
        t = np.arange(120)
        a = 0.01 * t + np.random.randn(120) * 0.1
        b = np.zeros_like(a)
        b[3:] = a[:-3]
        sector.create("a_series", a.tolist())
        sector.create("b_series", b.tolist())
        result = wave_ops.correlate("a_series", "b_series")
        assert "r" in result
        assert "lag" in result

    def test_decompose_unknown_sector(self):
        with pytest.raises(ValueError, match="not found"):
            wave_ops.decompose("nonexistent")

    def test_correlate_unknown_sector(self):
        sector.create("a_series", [1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="not found"):
            wave_ops.correlate("a_series", "nonexistent")


# ===================================================================
# template
# ===================================================================

class TestTemplate:
    def _make_sector_with_period(self, name: str, period: int,
                                  n_months: int = 120):
        t = np.arange(n_months)
        signal = np.sin(2 * np.pi * t / period) + np.random.randn(n_months) * 0.1
        plan = np.full(n_months, 100.0)
        sector.create(name, signal.tolist(), plan.tolist())

    def test_register_and_match(self):
        # Register a template (period 24 is a harmonic of 12 and will be
        # detected now that max_lag=36 in dominant_periods)
        template.register("boom", {
            "periods": [6, 12, 24],
            "amplitudes": [0.3, 0.2, 0.15],
            "mean_deviation": -0.02,
        })
        # Create a sector with period 12
        self._make_sector_with_period("test_sec", period=12)
        # Match should find some similarity (not None)
        result = template.match("test_sec")
        assert result["best_match"] is not None
        assert result["similarity"] > 0

    def test_match_no_match(self):
        # Register a template with very different signature
        template.register("flat", {
            "periods": [],
            "amplitudes": [],
            "mean_deviation": 0.0,
        })
        # Create sector with strong period
        self._make_sector_with_period("test_sec", period=12)
        result = template.match("test_sec")
        # With empty periods and zero mean deviation, similarity is below
        # threshold → best_match should be None
        assert result["best_match"] is None

    def test_list_templates(self):
        template.register("a", {"periods": [6], "amplitudes": [0.5], "mean_deviation": 0.0})
        template.register("b", {"periods": [12], "amplitudes": [0.3], "mean_deviation": -0.1})
        templates = template.list_templates()
        assert len(templates) == 2
        names = [t["name"] for t in templates]
        assert "a" in names
        assert "b" in names

    def test_list_templates_empty_library(self):
        assert template.list_templates("nonexistent") == []

    def test_register_duplicate_raises(self):
        template.register("dup", {"periods": [6], "amplitudes": [0.5], "mean_deviation": 0.0})
        with pytest.raises(ValueError, match="already exists"):
            template.register("dup", {"periods": [12], "amplitudes": [0.3], "mean_deviation": 0.0})

    def test_match_unknown_sector(self):
        # Register a template so library is non-empty
        template.register("dummy", {"periods": [6], "amplitudes": [0.5],
                                     "mean_deviation": 0.0})
        with pytest.raises(ValueError, match="not found"):
            template.match("nonexistent")

    def test_match_empty_library(self):
        t = np.arange(60)
        signal = np.sin(2 * np.pi * t / 12)
        sector.create("test_sec", signal.tolist())
        result = template.match("test_sec")
        assert result["best_match"] is None
        assert result["all_scores"] == {}


# ===================================================================
# branch
# ===================================================================

class TestBranch:
    def _make_sector_with_targets(self, name: str, n_months: int = 60):
        t = np.arange(n_months)
        signal = np.sin(2 * np.pi * t / 12) + 100.0
        plan = np.full(n_months, 100.0)
        sector.create(name, signal.tolist(), plan.tolist())

    def test_simulate_single_sector(self):
        self._make_sector_with_targets("coal")
        result = branch.simulate({"coal": -0.05}, n=5)
        assert len(result["branches"]) == 5
        for b in result["branches"]:
            assert b["sector_id"] == "coal"
            assert b["delta"] >= 0

    def test_simulate_multiple_sectors(self):
        self._make_sector_with_targets("coal")
        self._make_sector_with_targets("steel")
        result = branch.simulate({"coal": -0.05, "steel": 0.1}, n=5)
        assert len(result["branches"]) == 10  # 5 branches * 2 sectors
        sector_ids = {b["sector_id"] for b in result["branches"]}
        assert sector_ids == {"coal", "steel"}

    def test_branch_ensemble_stats(self):
        self._make_sector_with_targets("coal")
        result = branch.simulate({"coal": 0.0}, n=10)
        stats = result["ensemble_stats"]
        assert "mean_delta" in stats
        assert "std_delta" in stats
        assert "n_positive" in stats
        assert "n_negative" in stats

    def test_nodes_and_antinodes(self):
        self._make_sector_with_targets("coal")
        result = branch.simulate({"coal": 0.0}, n=10)
        assert isinstance(result["nodes"], list)
        assert isinstance(result["antinodes"], list)

    def test_unknown_sector_raises(self):
        with pytest.raises(ValueError, match="not found"):
            branch.simulate({"nonexistent": -0.1}, n=5)

    def test_sector_without_targets_raises(self):
        sector.create("no_targets", [1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="no targets"):
            branch.simulate({"no_targets": -0.1}, n=5)


# ===================================================================
# cluster
# ===================================================================

class TestCluster:
    def _signature(self, periods, amplitudes, mean_dev):
        return {"periods": periods, "amplitudes": amplitudes,
                "mean_deviation": mean_dev}

    def test_kmeans_separates(self):
        # Two distinct clusters
        sigs = [
            self._signature([6, 12], [0.5, 0.3], -0.02),
            self._signature([6, 12], [0.5, 0.3], -0.02),
            self._signature([6, 12], [0.5, 0.3], -0.02),
            self._signature([24, 36], [0.8, 0.6], 0.05),
            self._signature([24, 36], [0.8, 0.6], 0.05),
            self._signature([24, 36], [0.8, 0.6], 0.05),
        ]
        result = cluster.run(sigs, n_clusters=2)
        labels = result["labels"]
        # First 3 should be in one cluster, last 3 in another
        first = set(labels[i] for i in range(3))
        last = set(labels[i] for i in range(3, 6))
        assert len(first) == 1, f"First 3 not in same cluster: {labels}"
        assert len(last) == 1, f"Last 3 not in same cluster: {labels}"
        assert first != last, "Both groups assigned to same cluster"

    def test_kmeans_convergence(self):
        sigs = [self._signature([6 + i], [0.5], 0.0) for i in range(10)]
        result = cluster.run(sigs, n_clusters=3)
        assert result["n_iter"] < 100

    def test_kmeans_stability(self):
        sigs = [self._signature([6 + i], [0.5], 0.0) for i in range(10)]
        result = cluster.run(sigs, n_clusters=3)
        # With 3 init attempts, stability should exist
        if result["stability"] is not None:
            assert 0 <= result["stability"] <= 1.0

    def test_empty_signatures(self):
        result = cluster.run([], n_clusters=3)
        assert result["labels"] == {}
        assert result["centroids"] == []
        assert result["inertia"] == 0.0

    def test_single_signature(self):
        sigs = [self._signature([6, 12], [0.5, 0.3], -0.02)]
        result = cluster.run(sigs, n_clusters=5)
        assert len(result["labels"]) == 1
        assert result["labels"][0] == 0

    def test_kmeans_labels_dict_type(self):
        sigs = [self._signature([6], [0.5], 0.0) for _ in range(5)]
        result = cluster.run(sigs, n_clusters=2)
        assert all(isinstance(k, int) for k in result["labels"])
        assert all(isinstance(v, int) for v in result["labels"].values())


# ===================================================================
# verify
# ===================================================================

class TestVerify:
    def test_verify_asserted(self):
        """MAPE < threshold/2 → ASSERTED."""
        result = verify.verify("pred_1",
                               realized=[100.0, 102.0, 101.0],
                               predicted=[100.5, 101.5, 101.5])
        assert result["verdict"] == Verdict.ASSERTED
        assert result["re_calibrate"] is False

    def test_verify_contested(self):
        """MAPE between threshold/2 and threshold → CONTESTED."""
        # Create prediction with ~10% error
        result = verify.verify("pred_2",
                               realized=[100.0, 110.0, 105.0],
                               predicted=[90.0, 100.0, 95.0])
        assert result["verdict"] == Verdict.CONTESTED

    def test_verify_unknown(self):
        """MAPE >= threshold → UNKNOWN."""
        result = verify.verify("pred_3",
                               realized=[100.0, 200.0, 50.0],
                               predicted=[100.0, 100.0, 100.0])
        assert result["verdict"] == Verdict.UNKNOWN
        assert result["re_calibrate"] is True

    def test_state_log_append(self):
        verify.verify("pred_4", [1.0, 2.0], [1.1, 2.1])
        assert len(verify._state_log) == 1

    def test_state_log_retrieval(self):
        verify.verify("pred_a", [1.0, 2.0], [1.1, 2.1])
        verify.verify("pred_b", [3.0, 4.0], [2.9, 4.1])
        entries = verify.get_state_log(10)
        assert len(entries) == 2
        assert entries[0]["prediction_id"] == "pred_a"
        assert entries[1]["prediction_id"] == "pred_b"

    def test_clear_state_log(self):
        verify.verify("pred_x", [1.0], [1.1])
        verify.clear_state_log()
        assert len(verify._state_log) == 0

    def test_verify_empty_arrays(self):
        result = verify.verify("empty", [], [])
        assert result["mae"] == 0.0
        assert result["mape"] == 0.0
        assert result["verdict"] == Verdict.ASSERTED

    def test_verify_no_error(self):
        """Perfect prediction → ASSERTED, MAPE=0."""
        result = verify.verify("perfect",
                               realized=[100.0, 200.0, 300.0],
                               predicted=[100.0, 200.0, 300.0])
        assert result["mape"] == 0.0
        assert result["verdict"] == Verdict.ASSERTED
