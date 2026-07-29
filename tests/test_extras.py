"""Tests for spectrum_os.extras (FFT + PCA) and CLI/quantum wiring."""

import json
import math

import numpy as np

from spectrum_os.extras import fft_decompose, pca_decompose
from spectrum_os.kernel import osc
from spectrum_os.kernel._types import Verdict
from spectrum_os import cli


# ---------------------------------------------------------------------------
# FFT
# ---------------------------------------------------------------------------

class TestFFT:
    def test_sine_period_detected(self):
        n = 120
        t = np.arange(n)
        x = np.sin(2 * np.pi * t / 12.0)
        res = fft_decompose(x)
        top = res.values["peaks"][0]
        assert abs(top["period_months"] - 12.0) < 0.5
        assert res.verdict == Verdict.ASSERTED

    def test_phase_difference_recovered(self):
        n = 96
        t = np.arange(n)
        shift = math.pi / 3
        x1 = np.sin(2 * np.pi * t / 16.0)
        x2 = np.sin(2 * np.pi * t / 16.0 + shift)
        p1 = fft_decompose(x1).values["peaks"][0]["phase_angle"]
        p2 = fft_decompose(x2).values["peaks"][0]["phase_angle"]
        diff = (p2 - p1 + np.pi) % (2 * np.pi) - np.pi
        assert abs(diff - shift) < 0.2

    def test_short_and_constant_unknown(self):
        assert fft_decompose([1.0, 2.0, 3.0]).verdict == Verdict.UNKNOWN
        assert fft_decompose([5.0] * 50).verdict == Verdict.UNKNOWN

    def test_dominant_periods_rounded(self):
        t = np.arange(96)
        x = np.sin(2 * np.pi * t / 8.0)
        res = fft_decompose(x)
        assert 8 in res.dominant_periods


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

class TestPCA:
    def test_correlated_first_component(self):
        rng = np.random.default_rng(7)
        base = np.sin(np.linspace(0, 6, 100))
        data = {
            "a": base + rng.normal(0, 0.05, 100),
            "b": base * 1.5 + rng.normal(0, 0.05, 100),
            "c": base * 0.5 + rng.normal(0, 0.05, 100),
        }
        res = pca_decompose(data)
        first = res.values["components"][0]["explained_variance_ratio"]
        assert first > 0.8
        assert res.verdict == Verdict.ASSERTED

    def test_dict_and_ndarray_inputs(self):
        X = np.random.default_rng(1).normal(size=(3, 50))
        r1 = pca_decompose(X)
        r2 = pca_decompose({"x": X[0], "y": X[1], "z": X[2]})
        assert r1.values["components"][0]["explained_variance_ratio"] > 0
        assert r2.values["components"][0]["explained_variance_ratio"] > 0

    def test_degenerate_inputs(self):
        assert pca_decompose({}).verdict == Verdict.UNKNOWN
        assert pca_decompose({"only": [1.0] * 20}).verdict == Verdict.UNKNOWN
        assert pca_decompose(np.ones(5)).verdict == Verdict.UNKNOWN


# ---------------------------------------------------------------------------
# osc.quantum wiring
# ---------------------------------------------------------------------------

class TestQuantumWiring:
    def test_osc_quantum_property(self):
        q = osc.quantum
        assert hasattr(q, "tomography")
        assert hasattr(q, "decoherence_watch")
        assert hasattr(q, "kl_divergence")


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------

class TestCLI:
    def _make_store(self, tmp_path):
        store = tmp_path / "sectors.json"
        t = np.arange(60)
        data = {
            "sine12": {
                "id": "sine12",
                "name": "sine12",
                "timeseries": (np.sin(2 * np.pi * t / 12.0) + 10).tolist(),
                "targets": None,
                "created": "2026-07-24",
                "meta": None,
            },
            "const": {
                "id": "const",
                "name": "const",
                "timeseries": [1.0] * 60,
                "targets": None,
                "created": "2026-07-24",
                "meta": None,
            },
        }
        store.write_text(json.dumps(data))
        return store

    def test_decompose_human_and_json(self, tmp_path, capsys):
        store = self._make_store(tmp_path)
        rc = cli.main(["decompose", "sine12", "--store", str(store)])
        out = capsys.readouterr().out
        assert "sine12" in out
        assert rc in (0, 2)
        rc = cli.main(["decompose", "sine12", "--store", str(store), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["sector"] == "sine12"
        assert payload["verdict"] in ("asserted", "contested", "unknown")

    def test_decompose_unknown_exit_code(self, tmp_path, capsys):
        store = self._make_store(tmp_path)
        rc = cli.main(["decompose", "const", "--store", str(store), "--json"])
        capsys.readouterr()
        assert rc == 2

    def test_correlate(self, tmp_path, capsys):
        store = self._make_store(tmp_path)
        rc = cli.main(["correlate", "sine12", "sine12", "--store", str(store)])
        assert rc == 0
        assert "sine12" in capsys.readouterr().out

    def test_verify(self, tmp_path, capsys):
        r = tmp_path / "r.json"
        p = tmp_path / "p.json"
        r.write_text(json.dumps([1.0, 2.0, 3.0, 4.0]))
        p.write_text(json.dumps([1.1, 1.9, 3.1, 3.9]))
        rc = cli.main(["verify", "demo", "--realized", str(r), "--predicted", str(p)])
        out = capsys.readouterr().out
        assert "mape" in out
        assert rc in (0, 2)
