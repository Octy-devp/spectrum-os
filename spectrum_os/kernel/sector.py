"""Sector data management — in-memory dict + JSON serialization."""

import json
from datetime import datetime
from ._types import Sector

# In-memory store
_sectors: dict[str, Sector] = {}
_store_path: str | None = None

# Required meta keys when registering a synthetic sector (PLAN-23 §7.3
# guardrail: 標記優先於生成 — synthetic registration without full
# provenance is refused).
_SYNTHETIC_REQUIRED_META = ("anchors", "generated_by", "seed")


def _validate_meta(meta: dict | None) -> None:
    """Enforce synthetic provenance completeness.

    If ``meta["synthetic"]`` is true, the meta dict must also carry
    ``anchors``, ``generated_by`` and ``seed`` — otherwise the registration
    is rejected.  Non-synthetic meta passes through unchanged.
    """
    if not meta:
        return
    if meta.get("synthetic") is True:
        missing = [k for k in _SYNTHETIC_REQUIRED_META if k not in meta]
        if missing:
            raise ValueError(
                f"synthetic sector meta missing required keys: {missing} "
                f"(need {list(_SYNTHETIC_REQUIRED_META)})"
            )


def create(name: str, timeseries: list[float], targets: list[float] | None = None,
           meta: dict | None = None) -> Sector:
    """Register a new sector. ID is auto-generated from name (lowercase, no spaces)."""
    _validate_meta(meta)
    sector_id = name.lower().replace(" ", "_").replace("-", "_")
    sector = Sector(
        id=sector_id,
        name=name,
        timeseries=timeseries,
        targets=targets,
        created=datetime.now().isoformat(),
        meta=meta,
    )
    _sectors[sector_id] = sector
    return sector


def list_sectors() -> list[Sector]:
    """Return all registered sectors."""
    return list(_sectors.values())


def get(sector_id: str) -> Sector | None:
    """Get a sector by ID."""
    return _sectors.get(sector_id)


def save(path: str | None = None):
    """Serialize all sectors to JSON."""
    p = path or _store_path
    if p is None:
        raise ValueError("No path specified and no store path configured")
    data = {
        sid: {
            "id": s.id,
            "name": s.name,
            "timeseries": s.timeseries,
            "targets": s.targets,
            "created": s.created,
            "meta": s.meta,
        }
        for sid, s in _sectors.items()
    }
    with open(p, "w") as f:
        json.dump(data, f, indent=2)


def load(path: str):
    """Load sectors from JSON file."""
    global _store_path
    with open(path, "r") as f:
        data = json.load(f)
    _sectors.clear()
    for sid, sdata in data.items():
        _sectors[sid] = Sector(
            id=sdata["id"],
            name=sdata["name"],
            timeseries=sdata["timeseries"],
            targets=sdata.get("targets"),
            created=sdata["created"],
            meta=sdata.get("meta"),
        )
    _store_path = path
