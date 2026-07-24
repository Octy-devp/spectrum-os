"""Sector data management — in-memory dict + JSON serialization."""

import json
from datetime import datetime
from ._types import Sector

# In-memory store
_sectors: dict[str, Sector] = {}
_store_path: str | None = None


def create(name: str, timeseries: list[float], targets: list[float] | None = None) -> Sector:
    """Register a new sector. ID is auto-generated from name (lowercase, no spaces)."""
    sector_id = name.lower().replace(" ", "_").replace("-", "_")
    sector = Sector(
        id=sector_id,
        name=name,
        timeseries=timeseries,
        targets=targets,
        created=datetime.now().isoformat(),
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
        )
    _store_path = path
