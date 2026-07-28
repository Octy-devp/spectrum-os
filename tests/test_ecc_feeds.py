"""Tests for feeds/ecc_feeds.py — PLAN-23 §四-B 真實資料 feed。

分兩層：
- 結構不變量（長度/連續性/無 NaN/密度/決定性/ha 對源逐點一致）——永遠應通過。
- 快照契約（目前 verdict 值）——ECC 上游數據更新時預期會變，變了請人工確認後更新期望值。
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from feeds import ecc_feeds  # noqa: E402
from spectrum_os import osc  # noqa: E402
from spectrum_os.kernel._types import Verdict  # noqa: E402

EXPECTED = {
    #                    start       end        n   min_density
    "hk_mood":      ("1907-01", "1908-01",  13, 0.50),
    "germany_mood": ("1908-01", "1909-03",  15, 0.75),
    "return_mood":  ("1914-02", "1920-01",  72, 0.30),
    "ha_tension":   ("1900-01", "1920-12", 252, 1.00),
}


@pytest.fixture(scope="module")
def built():
    return ecc_feeds.build_all()


# ---------------------------------------------------------------------------
# 結構不變量
# ---------------------------------------------------------------------------

class TestSeriesStructure:
    def test_expected_grids(self, built):
        for sid, (start, end, n, _) in EXPECTED.items():
            b = built[sid]
            assert b["months"][0] == start, sid
            assert b["months"][-1] == end, sid
            assert len(b["values"]) == n, sid
            assert len(b["months"]) == n, sid

    def test_no_nan_all_finite(self, built):
        import math
        for sid, b in built.items():
            assert all(isinstance(v, float) and math.isfinite(v) for v in b["values"]), sid

    def test_grid_contiguous(self, built):
        for sid, b in built.items():
            idx = [ecc_feeds._month_index(m) for m in b["months"]]
            assert all(j - i == 1 for i, j in zip(idx, idx[1:])), sid

    def test_density_thresholds(self, built):
        for sid, (_, _, _, min_density) in EXPECTED.items():
            assert built[sid]["stats"]["density"] >= min_density, sid

    def test_observed_plus_filled_equals_grid(self, built):
        for sid, b in built.items():
            st = b["stats"]
            assert st["observed_months"] + st["filled_months"] == st["grid_months"], sid

    def test_grid_endpoints_are_observed_months(self, built):
        for sid in ("hk_mood", "germany_mood", "return_mood"):
            obs = built[sid]["monthly_observations"]
            months = built[sid]["months"]
            assert months[0] in obs and months[-1] in obs, sid

    def test_determinism(self, built):
        again = ecc_feeds.build_all()
        for sid in built:
            assert built[sid]["months"] == again[sid]["months"], sid
            assert built[sid]["values"] == again[sid]["values"], sid


class TestGapPolicy:
    def test_hk_drops_isolated_1900_11(self, built):
        dropped = built["hk_mood"]["stats"]["dropped_segments"]
        assert any(d["start"] == "1900-11" and d["observed_months"] == 1 for d in dropped)

    def test_no_fill_run_exceeds_limit(self, built):
        for sid, b in built.items():
            assert b["stats"]["longest_fill_run"] <= ecc_feeds.MAX_FILL_RUN, sid

    def test_ha_never_filled(self, built):
        assert built["ha_tension"]["stats"]["filled_months"] == 0
        assert built["ha_tension"]["stats"]["density"] == 1.0


class TestHaBaseline:
    """ha_tension 必須與 historical-atmosphere.json 的 tension_level 逐點一致。"""

    def test_matches_source_exactly(self, built):
        src = json.load(open(ecc_feeds.HISTORICAL_ATMOSPHERE, encoding="utf-8"))["time_series"]
        expected = [float(m["tension_level"]) for m in src]
        assert built["ha_tension"]["values"] == expected
        assert built["ha_tension"]["months"] == [m["date"] for m in src]


# ---------------------------------------------------------------------------
# 註冊 / 保存 / decompose
# ---------------------------------------------------------------------------

class TestRegisterAndDecompose:
    def test_register_save_load_roundtrip(self, built, tmp_path):
        ids = ecc_feeds.register_sectors(built)
        assert ids == list(EXPECTED)
        store = tmp_path / "sectors.json"
        osc.sectors.save(str(store))
        osc.sectors.load(str(store))
        for sid, b in built.items():
            sec = osc.sectors.get(sid)
            assert sec is not None, sid
            assert sec.timeseries == list(b["values"]), sid
            assert sec.targets is None, sid

    def test_decompose_returns_ternary_verdict(self, built):
        ecc_feeds.register_sectors(built)
        for sid, (_, _, n, _) in EXPECTED.items():
            dec = ecc_feeds.decompose_sector(sid)
            assert dec["verdict"] in {v.value for v in Verdict}, sid
            assert isinstance(dec["dominant_periods"], list), sid
            lw, mw = dec["long_window"], dec["mid_window"]
            assert (lw, mw) == ((36, 6) if n >= 48 else (max(3, n // 2), max(2, max(3, n // 2) // 3))), sid

    def test_verdict_snapshot(self, built):
        """快照契約：2026-07-28 首次餵真實數據的 verdict。

        return_mood 的 ASSERTED 是 ffill 高原抬升短 lag 自相關的上界樂觀結果
        （density=0.33 且 dominant_periods=[]，無週期結構）——判讀見 feed_report。
        上游數據變動時此測試預期失敗，需人工確認後更新。
        """
        ecc_feeds.register_sectors(built)
        snap = {sid: ecc_feeds.decompose_sector(sid)["verdict"] for sid in EXPECTED}
        assert snap == {
            "hk_mood": "contested",
            "germany_mood": "unknown",
            "return_mood": "asserted",
            "ha_tension": "contested",
        }


class TestMainArtifacts:
    def test_main_writes_sectors_and_report(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ecc_feeds, "SECTORS_DIR", tmp_path)
        assert ecc_feeds.main() == 0
        store = json.load(open(tmp_path / "sectors.json", encoding="utf-8"))
        assert set(store) == set(EXPECTED)
        report = json.load(open(tmp_path / "feed_report.json", encoding="utf-8"))
        for sid in EXPECTED:
            entry = report["sectors"][sid]
            assert entry["series"]["stats"]["density"] > 0
            assert entry["decompose"]["verdict"] in {v.value for v in Verdict}
