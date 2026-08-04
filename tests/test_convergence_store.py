"""Tests for synth/convergence_store.py — T19 companion 收束持久層。

仿 warfare dashboard 模式：指向哪就寫到哪（非 hardcode）+ .bak 自動備份 +
每輪快照。全部 tmp_path 落地，零真 API。世界無關檢定：模組產物不得含世界特定詞。
"""

from __future__ import annotations

import json

import pytest

from spectrum_os.contracts import FieldLog
from spectrum_os.synth.convergence_store import (
    list_rounds,
    load_field_log,
    load_round_snapshot,
    save_dashboard,
    save_field_log,
    save_round_snapshot,
)

#: 世界無關檢定禁詞（出現即違規）。
WORLD_WORDS = [
    "ECC", "ecc", "巴爾幹", "1914", "沙俄", "德意志", "蘇維埃", "奧匈",
    "塞爾維亞", "薩拉熱窩", "hongkong", "香港",
]


def _make_field_log() -> FieldLog:
    fl = FieldLog(field_id="f-test")
    fl.add_entry(
        1,
        {"label": "route-a", "grounding": "rail mismatch"},
        [{"label": "route-b"}, {"label": "route-c"}],
    )
    return fl


def _make_round_result() -> dict:
    """probe_converge_round 回傳的最小形狀（僅含快照需要的鍵）。"""
    return {
        "view": "[root] situation root\n└─ L1 route-a（已選）\n└─ L1 route-b（未選）",
        "collapse": {"selected": "route-a", "layer": 1, "unselected": ["route-b"]},
        "path": ["route-a"],
        "field_log_entry": {
            "layer": 1,
            "selected": {"label": "route-a"},
            "unselected": [{"label": "route-b"}],
        },
        "state_log_entry": {"gate_type": "probe_tree", "selected_branch": "route-a"},
    }


# ---------------------------------------------------------------------------
# 1. save_dashboard：視圖落盤
# ---------------------------------------------------------------------------

class TestSaveDashboard:
    def test_writes_view_to_explicit_path(self, tmp_path):
        out = tmp_path / "rounds" / "round-1-dashboard.md"
        view = "[root] situation root\n└─ L1 route-a（可選）"
        written = save_dashboard(view, out)
        assert written == out
        assert out.read_text(encoding="utf-8") == view
        # 目錄自動建立
        assert out.parent.exists()

    def test_backup_previous_version(self, tmp_path):
        out = tmp_path / "dashboard.md"
        save_dashboard("version-1", out)
        save_dashboard("version-2", out)
        # 舊版自動備份為 .bak（仿 save_oob_to_profiles）
        bak = tmp_path / "dashboard.md.bak"
        assert bak.read_text(encoding="utf-8") == "version-1"
        assert out.read_text(encoding="utf-8") == "version-2"

    def test_rejects_non_str_view(self, tmp_path):
        with pytest.raises(TypeError, match="view"):
            save_dashboard({"trees": []}, tmp_path / "x.md")


# ---------------------------------------------------------------------------
# 2. FieldLog 落盤 / 讀回
# ---------------------------------------------------------------------------

class TestFieldLogStore:
    def test_save_load_round_trip(self, tmp_path):
        fl = _make_field_log()
        p = tmp_path / "field-log.json"
        save_field_log(fl, p)
        loaded = load_field_log(p)
        assert isinstance(loaded, FieldLog)
        assert loaded.field_id == "f-test"
        assert loaded.to_dict() == fl.to_dict()
        # unselected 保留（維持疊加，不刪除）
        entry = loaded.entries[0]
        assert entry["selected"]["label"] == "route-a"
        assert {b["label"] for b in entry["unselected"]} == {"route-b", "route-c"}

    def test_append_accumulates_entries(self, tmp_path):
        p = tmp_path / "field-log.json"
        fl = FieldLog(field_id="f-test")
        fl.add_entry(1, {"label": "a"}, [{"label": "b"}])
        save_field_log(fl, p)
        # 續接：載入 → 加一筆 → 再存
        fl2 = load_field_log(p)
        fl2.add_entry(2, {"label": "a1"}, [{"label": "a2"}])
        save_field_log(fl2, p)
        fl3 = load_field_log(p)
        assert len(fl3.entries) == 2
        assert fl3.entries[1]["selected"]["label"] == "a1"

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_field_log(tmp_path / "nope.json")

    def test_load_invalid_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"entries": []}), encoding="utf-8")  # 缺 field_id
        with pytest.raises(ValueError, match="field_id"):
            load_field_log(p)

    def test_rejects_non_fieldlog(self, tmp_path):
        with pytest.raises(TypeError, match="FieldLog"):
            save_field_log({"entries": []}, tmp_path / "x.json")

    def test_backup_previous_version(self, tmp_path):
        p = tmp_path / "field-log.json"
        fl1 = FieldLog(field_id="f-test")
        fl1.add_entry(1, {"label": "a"}, [])
        save_field_log(fl1, p)
        fl2 = load_field_log(p)
        fl2.add_entry(2, {"label": "b"}, [])
        save_field_log(fl2, p)
        bak = tmp_path / "field-log.json.bak"
        assert load_field_log(bak).entries[0]["selected"]["label"] == "a"


# ---------------------------------------------------------------------------
# 3. round snapshot：每輪快照 + 跨 session 續接
# ---------------------------------------------------------------------------

class TestRoundSnapshot:
    def test_save_load_round_trip(self, tmp_path):
        result = _make_round_result()
        written = save_round_snapshot(result, 1, tmp_path)
        assert written.name == "converge-round-01.json"
        snap = load_round_snapshot(1, tmp_path)
        assert snap["round"] == 1
        assert snap["collapse"]["selected"] == "route-a"
        assert snap["path"] == ["route-a"]
        assert snap["view"] == result["view"]
        assert snap["field_log_entry"]["selected"]["label"] == "route-a"
        assert snap["state_log_entry"]["selected_branch"] == "route-a"

    def test_lightweight_only_whitelisted_keys(self, tmp_path):
        result = _make_round_result()
        result["expanded"] = {"huge": "payload"}  # 不應落盤
        result["next_state"] = {"layers": ["..."]}  # 不應落盤
        save_round_snapshot(result, 1, tmp_path)
        snap = load_round_snapshot(1, tmp_path)
        assert "expanded" not in snap
        assert "next_state" not in snap
        assert set(snap.keys()) <= {
            "round", "view", "collapse", "path", "field_log_entry", "state_log_entry",
        }

    def test_multiple_rounds_listed(self, tmp_path):
        save_round_snapshot(_make_round_result(), 1, tmp_path)
        save_round_snapshot(_make_round_result(), 2, tmp_path)
        save_round_snapshot(_make_round_result(), 3, tmp_path)
        assert list_rounds(tmp_path) == [1, 2, 3]

    def test_list_empty_dir(self, tmp_path):
        assert list_rounds(tmp_path) == []

    def test_load_missing_round_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_round_snapshot(5, tmp_path)

    def test_backup_previous_snapshot(self, tmp_path):
        save_round_snapshot(_make_round_result(), 1, tmp_path)
        # 覆寫同輪 → .bak 保留舊版
        save_round_snapshot(_make_round_result(), 1, tmp_path)
        bak = tmp_path / "converge-round-01.json.bak"
        assert bak.exists()

    def test_invalid_round_num_raises(self, tmp_path):
        with pytest.raises(ValueError, match="round_num"):
            save_round_snapshot(_make_round_result(), 0, tmp_path)

    def test_rejects_non_dict_result(self, tmp_path):
        with pytest.raises(TypeError, match="round_result"):
            save_round_snapshot(["not", "dict"], 1, tmp_path)


# ---------------------------------------------------------------------------
# 4. 世界無關檢定
# ---------------------------------------------------------------------------

class TestWorldAgnostic:
    def test_written_files_contain_no_world_words(self, tmp_path):
        fl = _make_field_log()
        save_field_log(fl, tmp_path / "field-log.json")
        save_dashboard(
            "[root] situation root\n└─ L1 route-a（已選）", tmp_path / "dash.md"
        )
        save_round_snapshot(_make_round_result(), 1, tmp_path)
        for p in tmp_path.iterdir():
            if p.suffix == ".json" or p.suffix == ".md":
                text = p.read_text(encoding="utf-8")
                for word in WORLD_WORDS:
                    assert word not in text, f"{p.name} 含世界特定詞 {word!r}"

    def test_snapshot_filename_world_agnostic(self, tmp_path):
        save_round_snapshot(_make_round_result(), 1, tmp_path)
        names = [p.name for p in tmp_path.iterdir()]
        assert "converge-round-01.json" in names  # 無世界名
