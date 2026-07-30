"""Tests for spectrum_os/sources.py & core generality assertions (PLAN-23 §6.6)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from spectrum_os import osc
from spectrum_os.contracts import CLADCorpus, DCASubstrate, RoleTrajectory
from spectrum_os.sources import (
    SourceSpec,
    clear_registry,
    get_source,
    list_sources,
    load_source,
    register_source,
)


@pytest.fixture(autouse=True)
def clean_registry():
    clear_registry()
    yield
    clear_registry()


class TestSourceRegistry:
    def test_register_get_list(self):
        spec = SourceSpec(
            driver="file",
            locator="dummy.json",
            emits="RoleTrajectory",
            source_id="test_src_1",
        )
        sid = register_source(spec)
        assert sid == "test_src_1"
        assert list_sources() == ["test_src_1"]
        assert get_source("test_src_1") == spec

    def test_get_nonexistent_raises_keyerror(self):
        with pytest.raises(KeyError, match="Source not found"):
            get_source("nonexistent")

    def test_unknown_driver_raises_valueerror(self):
        spec = SourceSpec(
            driver="invalid_driver",
            locator="dummy.json",
            emits="RoleTrajectory",
            source_id="bad_src",
        )
        register_source(spec)
        with pytest.raises(ValueError, match="Unknown driver"):
            load_source("bad_src")


class TestFileDriver:
    def test_load_json_role_trajectory(self, tmp_path):
        data = {
            "thread_id": "thread_abc",
            "points": [[1.0, "crisis"], [2.0, "lag"]],
        }
        json_file = tmp_path / "traj.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        spec = SourceSpec(
            driver="file",
            locator=str(json_file),
            emits="RoleTrajectory",
            source_id="file_traj",
        )
        register_source(spec)

        traj = load_source("file_traj")
        assert isinstance(traj, RoleTrajectory)
        assert traj.thread_id == "thread_abc"
        assert traj.role_sequence() == ["crisis", "lag"]

    def test_load_yaml_dca_substrate(self, tmp_path):
        yaml_content = """
nodes:
  s1: {mass: 10}
  s2: {mass: 5}
edges:
  - from: s1
    to: s2
    relation: crisis->lag
    strength: 1.0
"""
        yaml_file = tmp_path / "sub.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        spec = SourceSpec(
            driver="file",
            locator=str(yaml_file),
            emits="DCASubstrate",
            source_id="file_dca",
        )
        register_source(spec)

        sub = load_source("file_dca")
        assert isinstance(sub, DCASubstrate)
        assert "s1" in sub.nodes
        assert len(sub.edges) == 1

    def test_load_jsonl_clad_corpus(self, tmp_path):
        jsonl_content = '{"id": "entry1", "clad": {"crisis": "c1", "lag": "l1", "alternative": "a1", "direction": "d1"}}\n'
        jsonl_file = tmp_path / "clad.jsonl"
        jsonl_file.write_text(jsonl_content, encoding="utf-8")

        spec = SourceSpec(
            driver="file",
            locator=str(jsonl_file),
            emits="CLADCorpus",
            source_id="file_clad",
        )
        register_source(spec)

        corpus = load_source("file_clad")
        assert isinstance(corpus, CLADCorpus)
        assert len(corpus.entries) == 1
        assert corpus.entries[0]["id"] == "entry1"


class TestHttpApiDriver:
    def test_http_api_worldbank_shape_with_caching(self, tmp_path, monkeypatch):
        # Redirect cache dir to tmp_path
        import spectrum_os.sources as sources_mod

        monkeypatch.setattr(sources_mod, "CACHE_DIR", tmp_path / "cache")

        mock_wb_response = [
            {"page": 1},
            [
                {"date": "2020", "value": 2.5},
                {"date": "2021", "value": 5.7},
            ],
        ]

        mock_urlopen = MagicMock()
        mock_urlopen.return_value.read.return_value = json.dumps(mock_wb_response).encode("utf-8")

        spec = SourceSpec(
            driver="http_api",
            locator="https://api.worldbank.org/v2/country/WLD/indicator/NY.GDP.MKTP.KD.ZG",
            emits="Sector",
            source_id="wb_world_gdp",
        )
        register_source(spec)

        with patch("urllib.request.urlopen", mock_urlopen):
            # First load: fetches via mock urlopen and writes cache
            sec1 = load_source("wb_world_gdp")
            assert sec1.id == "wb_world_gdp"
            assert sec1.timeseries == [2.5, 5.7]
            assert mock_urlopen.call_count == 1

            # Second load: must read from cache without calling urlopen again
            sec2 = load_source("wb_world_gdp")
            assert sec2.timeseries == [2.5, 5.7]
            assert mock_urlopen.call_count == 1  # Unchanged!


class TestLlmKnowledgeDriver:
    def test_llm_knowledge_delegate_call(self):
        mock_gate_fn = MagicMock()
        mock_gate_fn.return_value = {
            "entries": [
                {
                    "id": "e_synth",
                    "clad": {"crisis": "c", "lag": "l", "alternative": "a", "direction": "d"},
                }
            ]
        }

        spec = SourceSpec(
            driver="llm_knowledge",
            locator="synth_gate_v1",
            emits="CLADCorpus",
            meta={"gate_fn": mock_gate_fn},
            source_id="llm_clad",
        )
        register_source(spec)

        corpus = load_source("llm_clad")
        assert isinstance(corpus, CLADCorpus)
        assert len(corpus.entries) == 1
        assert corpus.entries[0]["id"] == "e_synth"
        assert mock_gate_fn.call_count == 1


class TestOscFacadeProperties:
    def test_osc_contracts_and_sources(self):
        assert osc.contracts is not None
        assert osc.sources is not None
        assert hasattr(osc.contracts, "RoleTrajectory")
        assert hasattr(osc.sources, "SourceSpec")


class TestCoreGeneralityMechanicalCheck:
    """CRITICAL: Ensure core modules contain NO hardcoded ECC absolute paths."""

    def test_no_ecc_paths_in_core_modules(self):
        repo_root = Path(__file__).resolve().parents[1]
        core_dirs_and_files = [
            repo_root / "spectrum_os" / "contracts.py",
            repo_root / "spectrum_os" / "sources.py",
            repo_root / "spectrum_os" / "kernel",
            repo_root / "spectrum_os" / "quantum",
        ]

        forbidden_literals = ["/home/octy/projects/ECC", "index/"]

        py_files: list[Path] = []
        for target in core_dirs_and_files:
            if target.is_file():
                py_files.append(target)
            elif target.is_dir():
                py_files.extend(target.rglob("*.py"))

        violations = []
        for py_file in py_files:
            content = py_file.read_text(encoding="utf-8")
            for forbidden in forbidden_literals:
                if forbidden in content:
                    violations.append((str(py_file.relative_to(repo_root)), forbidden))

        assert not violations, f"Forbidden ECC literals found in core modules: {violations}"
