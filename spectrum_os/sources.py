"""Spectrum OS Universal Source Registry & Drivers (PLAN-23 §6.6).

Provides SourceSpec registry and loaders for declarative data sources:
- file: generic JSON / YAML / JSONL loader with field mapping.
- http_api: generic HTTP JSON client with data/sources_cache/ caching.
- llm_knowledge: thin wrapper for gate configs and synthesis delegates.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Optional PyYAML support
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

CACHE_DIR = Path("data/sources_cache")


@dataclass
class SourceSpec:
    """Declarative specification for a data source (PLAN-23 §6.6)."""

    driver: str          # "http_api" | "file" | "llm_knowledge"
    locator: str         # url template / path / gate config id
    emits: str           # "Sector" | "RoleTrajectory" | "DCASubstrate" | "CLADCorpus"
    mapping: dict = field(default_factory=dict)
    bias_flags: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    source_id: str = ""


_SOURCES_REGISTRY: dict[str, SourceSpec] = {}


def register_source(spec: SourceSpec) -> str:
    """Register a SourceSpec in the global registry."""
    if not spec.source_id:
        spec.source_id = f"{spec.driver}-{spec.emits.lower()}-{len(_SOURCES_REGISTRY) + 1}"
    _SOURCES_REGISTRY[spec.source_id] = spec
    return spec.source_id


def list_sources() -> list[str]:
    """Return a sorted list of all registered source_ids."""
    return sorted(_SOURCES_REGISTRY.keys())


def get_source(source_id: str) -> SourceSpec:
    """Get a registered SourceSpec by source_id."""
    if source_id not in _SOURCES_REGISTRY:
        raise KeyError(f"Source not found in registry: {source_id!r}")
    return _SOURCES_REGISTRY[source_id]


def clear_registry() -> None:
    """Clear all registered sources."""
    _SOURCES_REGISTRY.clear()


def load_source(source_id_or_spec: str | SourceSpec, **kwargs: Any) -> Any:
    """Load data using driver specified in SourceSpec and return contract object."""
    if isinstance(source_id_or_spec, str):
        spec = get_source(source_id_or_spec)
    elif isinstance(source_id_or_spec, SourceSpec):
        spec = source_id_or_spec
    else:
        raise TypeError(f"Expected str or SourceSpec, got {type(source_id_or_spec)}")

    if spec.driver == "file":
        return _load_file_source(spec)
    elif spec.driver == "http_api":
        return _load_http_api_source(spec, **kwargs)
    elif spec.driver == "llm_knowledge":
        return _load_llm_knowledge_source(spec, **kwargs)
    else:
        raise ValueError(f"Unknown driver {spec.driver!r} in SourceSpec for {spec.source_id!r}")


def _load_file_source(spec: SourceSpec) -> Any:
    path = Path(spec.locator)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    ext = path.suffix.lower()
    if ext in (".yaml", ".yml"):
        if not _HAS_YAML:
            raise RuntimeError("PyYAML is not installed; cannot load YAML source files")
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    elif ext == ".jsonl":
        raw_data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_data.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

    return _build_contract_object(spec, raw_data)


def _load_http_api_source(spec: SourceSpec, url_args: dict | None = None) -> Any:
    locator = spec.locator
    if url_args:
        url = locator.format(**url_args)
    else:
        url = locator

    source_key = spec.source_id or "api_source"
    cache_file = CACHE_DIR / f"{source_key}.json"

    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    else:
        req = urllib.request.urlopen(url)
        raw_bytes = req.read()
        raw_data = json.loads(raw_bytes.decode("utf-8"))
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=2)

    return _build_contract_object(spec, raw_data)


def _load_llm_knowledge_source(spec: SourceSpec, delegate_fn: Callable | None = None) -> Any:
    fn = delegate_fn or spec.meta.get("gate_fn")
    if callable(fn):
        res = fn(spec)
        if isinstance(res, (dict, list)):
            return _build_contract_object(spec, res)
        return res

    raw_data = spec.meta.get("data", {})
    return _build_contract_object(spec, raw_data)


def _build_contract_object(spec: SourceSpec, raw_data: Any) -> Any:
    from spectrum_os.contracts import CLADCorpus, DCASubstrate, RoleTrajectory
    from spectrum_os.kernel import osc

    mapping = spec.mapping or {}
    emits = spec.emits

    if emits == "Sector":
        val_key = mapping.get("values", "values")
        if isinstance(raw_data, dict) and val_key in raw_data:
            values = raw_data[val_key]
        elif isinstance(raw_data, list):
            # Support World Bank style API response: [meta_dict, records_list]
            if len(raw_data) == 2 and isinstance(raw_data[0], dict) and isinstance(raw_data[1], list):
                records = raw_data[1]
                v_field = mapping.get("value_field", "value")
                values = [float(item[v_field]) for item in records if isinstance(item, dict) and v_field in item and item[v_field] is not None]
            else:
                values = raw_data
        else:
            values = raw_data

        if isinstance(values, dict):
            values = list(values.values())
        return osc.sectors.create(spec.source_id, timeseries=list(values))

    elif emits == "RoleTrajectory":
        if isinstance(raw_data, dict):
            thread_id = str(raw_data.get(mapping.get("thread_id", "thread_id"), spec.source_id))
            points = raw_data.get(mapping.get("points", "points"), [])
        elif isinstance(raw_data, list):
            thread_id = spec.source_id
            points = raw_data
        else:
            thread_id = spec.source_id
            points = []
        return RoleTrajectory(
            thread_id=thread_id,
            points=points,
            source_id=spec.source_id,
            bias_flags=spec.bias_flags,
            meta=spec.meta,
        )

    elif emits == "DCASubstrate":
        if isinstance(raw_data, dict):
            nodes = raw_data.get(mapping.get("nodes", "nodes"), {})
            edges = raw_data.get(mapping.get("edges", "edges"), [])
        else:
            nodes, edges = {}, []
        return DCASubstrate(
            nodes=nodes,
            edges=edges,
            source_id=spec.source_id,
            bias_flags=spec.bias_flags,
            meta=spec.meta,
        )

    elif emits == "CLADCorpus":
        if isinstance(raw_data, dict):
            entries = raw_data.get(mapping.get("entries", "entries"), [])
        elif isinstance(raw_data, list):
            entries = raw_data
        else:
            entries = []
        return CLADCorpus(
            entries=entries,
            source_id=spec.source_id,
            bias_flags=spec.bias_flags,
            meta=spec.meta,
        )

    else:
        raise ValueError(f"Unknown emit type {emits!r} in SourceSpec for {spec.source_id!r}")
