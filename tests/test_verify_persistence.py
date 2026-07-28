"""Tests for verify state_log JSONL persistence (PLAN-23 待辦 A).

Covers: init_log / append on verify() / reload across restart /
query filters / summary aggregation / 10k+ row query performance.
"""

import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from spectrum_os.kernel import verify
from spectrum_os.kernel._types import Verdict

LOG_FIELDS = {"ts", "prediction_id", "mae", "mape", "verdict", "re_calibrate"}


@pytest.fixture(autouse=True)
def reset_verify_state():
    """Isolate memory buffer and persistence path around each test."""
    verify._state_log.clear()
    verify._log_path = None
    yield
    verify._state_log.clear()
    verify._log_path = None


def _write_lines(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _make_record(i, prediction_id=None, verdict="asserted",
                 re_calibrate=False, mape=0.01, ts=None):
    return {
        "ts": ts or datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
        "prediction_id": prediction_id or f"pred_{i}",
        "mae": mape * 100.0,
        "mape": mape,
        "verdict": verdict,
        "re_calibrate": re_calibrate,
    }


# ===================================================================
# init_log + append
# ===================================================================

class TestInitLog:
    def test_init_creates_file_and_parents(self, tmp_path):
        target = tmp_path / "nested" / "data" / "state_log.jsonl"
        result = verify.init_log(target)
        assert result == target
        assert target.exists()
        assert verify._log_path == target

    def test_init_returns_path_and_default_arg(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = verify.init_log()
        assert result.name == "state_log.jsonl"
        assert result.parent.name == "data"
        assert result.exists()

    def test_verify_appends_jsonl_with_exact_fields(self, tmp_path):
        target = tmp_path / "state_log.jsonl"
        verify.init_log(target)
        verify.verify("pred_1", realized=[100.0, 102.0], predicted=[100.5, 101.5])
        lines = target.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert set(rec.keys()) == LOG_FIELDS
        assert rec["prediction_id"] == "pred_1"
        assert rec["verdict"] == "asserted"
        assert rec["re_calibrate"] is False
        # ts is valid ISO 8601
        datetime.fromisoformat(rec["ts"])

    def test_memory_buffer_still_updated_when_persisting(self, tmp_path):
        verify.init_log(tmp_path / "state_log.jsonl")
        verify.verify("pred_1", [1.0, 2.0], [1.1, 2.1])
        assert len(verify._state_log) == 1
        entry = verify._state_log[0]
        assert entry["prediction_id"] == "pred_1"
        assert "ts" in entry
        # memory entry keeps the full series; JSONL line does not
        assert entry["realized"] == [1.0, 2.0]
        assert entry["predicted"] == [1.1, 2.1]

    def test_no_persistence_without_init(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        verify.verify("pred_1", [1.0], [1.1])
        assert len(verify._state_log) == 1
        assert not (tmp_path / "data").exists()

    def test_init_none_disables_persistence(self, tmp_path):
        target = tmp_path / "state_log.jsonl"
        verify.init_log(target)
        verify.verify("pred_1", [1.0], [1.1])
        verify.init_log(None)
        assert verify._log_path is None
        verify.verify("pred_2", [1.0], [1.1])
        lines = target.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1  # only pred_1 was persisted

    def test_evaluate_alias_persists(self, tmp_path):
        target = tmp_path / "state_log.jsonl"
        verify.init_log(target)
        verify.evaluate("pred_alias", [1.0, 2.0], [1.0, 2.0])
        recs = verify.query(prediction_id="pred_alias")
        assert len(recs) == 1


# ===================================================================
# reload (restart durability)
# ===================================================================

class TestReload:
    def test_reinit_preserves_existing_file(self, tmp_path):
        target = tmp_path / "state_log.jsonl"
        verify.init_log(target)
        verify.verify("pred_a", [1.0], [1.1])
        verify.verify("pred_b", [1.0], [1.1])
        # simulate restart: fresh module state, same file
        verify._log_path = None
        verify._state_log.clear()
        verify.init_log(target)
        verify.verify("pred_c", [1.0], [1.1])
        recs = verify.query()
        assert [r["prediction_id"] for r in recs] == ["pred_a", "pred_b", "pred_c"]

    def test_query_via_explicit_path_without_init(self, tmp_path):
        target = tmp_path / "state_log.jsonl"
        verify.init_log(target)
        verify.verify("pred_a", [1.0], [1.1])
        verify._log_path = None  # simulate restart without init_log
        recs = verify.query(path=target)
        assert len(recs) == 1
        with pytest.raises(RuntimeError, match="init_log"):
            verify.query()


# ===================================================================
# query
# ===================================================================

class TestQuery:
    @pytest.fixture
    def log_file(self, tmp_path):
        """File with mixed verdicts / ids / timestamps."""
        target = tmp_path / "state_log.jsonl"
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        records = []
        for i in range(30):
            verdict = ["asserted", "contested", "unknown"][i % 3]
            records.append(_make_record(
                i,
                prediction_id=f"pred_{i % 5}",
                verdict=verdict,
                re_calibrate=(verdict == "unknown"),
                mape=0.01 * (i % 3 + 1),
                ts=(base + timedelta(hours=i)).isoformat(),
            ))
        _write_lines(target, records)
        verify.init_log(target)
        return target

    def test_query_all(self, log_file):
        assert len(verify.query()) == 30

    def test_query_by_prediction_id(self, log_file):
        recs = verify.query(prediction_id="pred_2")
        assert len(recs) == 6
        assert all(r["prediction_id"] == "pred_2" for r in recs)

    def test_query_by_verdict_string(self, log_file):
        recs = verify.query(verdict="unknown")
        assert len(recs) == 10
        assert all(r["verdict"] == "unknown" for r in recs)

    def test_query_by_verdict_enum(self, log_file):
        recs = verify.query(verdict=Verdict.CONTESTED)
        assert len(recs) == 10
        assert all(r["verdict"] == "contested" for r in recs)

    def test_query_time_range(self, log_file):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        recs = verify.query(ts_from=(base + timedelta(hours=5)).isoformat(),
                            ts_to=(base + timedelta(hours=9)).isoformat())
        assert len(recs) == 5  # hours 5..9 inclusive
        # date-only / naive bounds treated as UTC midnights
        recs2 = verify.query(ts_from="2026-01-02")
        assert len(recs2) == 6  # hours 24..29
        assert all(r["ts"].startswith("2026-01-02") for r in recs2)
        recs3 = verify.query(ts_from="2026-01-01", ts_to="2026-01-01")
        assert len(recs3) == 1  # only 2026-01-01T00:00 exactly

    def test_query_combined_filters(self, log_file):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        recs = verify.query(verdict="asserted",
                            ts_to=(base + timedelta(hours=10)).isoformat())
        assert all(r["verdict"] == "asserted" for r in recs)
        assert len(recs) == 4  # i = 0, 3, 6, 9

    def test_query_limit_returns_most_recent(self, log_file):
        recs = verify.query(verdict="asserted", limit=2)
        assert len(recs) == 2
        assert recs[-1]["prediction_id"] == "pred_2"  # i = 27
        assert recs[0]["prediction_id"] == "pred_4"   # i = 24

    def test_query_skips_corrupt_lines(self, log_file):
        with open(log_file, "a", encoding="utf-8") as fh:
            fh.write("not json\n\n")
        assert len(verify.query()) == 30

    def test_query_missing_file_returns_empty(self, tmp_path):
        assert verify.query(path=tmp_path / "nope.jsonl") == []


# ===================================================================
# summary
# ===================================================================

class TestSummary:
    def test_summary_aggregates(self, tmp_path):
        target = tmp_path / "state_log.jsonl"
        records = []
        for i in range(20):
            verdict = ["asserted", "contested", "contested", "unknown"][i % 4]
            records.append(_make_record(
                i, verdict=verdict,
                re_calibrate=(verdict == "unknown"),
                mape=float(i) / 100.0,
            ))
        _write_lines(target, records)
        verify.init_log(target)
        s = verify.summary(n=5)
        assert s["total"] == 20
        assert s["verdicts"] == {"asserted": 5, "contested": 10, "unknown": 5}
        assert s["re_calibrate_rate"] == pytest.approx(0.25)
        assert s["recent_n"] == 5
        # last 5 mapes: i=15..19 → 0.15..0.19
        assert s["recent_mape_mean"] == pytest.approx(0.17)

    def test_summary_empty_log(self, tmp_path):
        target = verify.init_log(tmp_path / "state_log.jsonl")
        s = verify.summary()
        assert s == {"total": 0, "verdicts": {}, "re_calibrate_rate": 0.0,
                     "recent_n": 0, "recent_mape_mean": 0.0}
        assert target.exists()

    def test_summary_via_explicit_path(self, tmp_path):
        target = tmp_path / "state_log.jsonl"
        _write_lines(target, [_make_record(1)])
        s = verify.summary(path=target)
        assert s["total"] == 1


# ===================================================================
# performance: 10k+ rows, query < 100 ms (PLAN-23 §十 成功條件 3)
# ===================================================================

class TestPerformance:
    def test_query_10k_under_100ms(self, tmp_path):
        target = tmp_path / "state_log.jsonl"
        n = 12_000
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with open(target, "w", encoding="utf-8") as fh:
            for i in range(n):
                fh.write(json.dumps(_make_record(
                    i,
                    prediction_id=f"pred_{i % 100}",
                    verdict=["asserted", "contested", "unknown"][i % 3],
                    re_calibrate=(i % 3 == 2),
                    mape=0.01 * (i % 3 + 1),
                    ts=(base + timedelta(minutes=i)).isoformat(),
                )) + "\n")
        verify.init_log(target)

        t0 = time.perf_counter()
        recs = verify.query(prediction_id="pred_42")
        id_elapsed = time.perf_counter() - t0
        assert len(recs) == n // 100
        assert id_elapsed < 0.1, f"query by prediction_id took {id_elapsed*1000:.1f} ms"

        t0 = time.perf_counter()
        recs = verify.query(verdict="unknown")
        verdict_elapsed = time.perf_counter() - t0
        assert len(recs) == n // 3
        assert verdict_elapsed < 0.1, f"query by verdict took {verdict_elapsed*1000:.1f} ms"

        t0 = time.perf_counter()
        recs = verify.query(ts_from=(base + timedelta(minutes=6000)).isoformat(),
                            ts_to=(base + timedelta(minutes=6999)).isoformat())
        ts_elapsed = time.perf_counter() - t0
        assert len(recs) == 1000
        assert ts_elapsed < 0.1, f"time-range query took {ts_elapsed*1000:.1f} ms"

    def test_summary_10k(self, tmp_path):
        target = tmp_path / "state_log.jsonl"
        n = 12_000
        with open(target, "w", encoding="utf-8") as fh:
            for i in range(n):
                fh.write(json.dumps(_make_record(
                    i, verdict="unknown" if i % 3 == 2 else "asserted",
                    re_calibrate=(i % 3 == 2), mape=0.05,
                )) + "\n")
        verify.init_log(target)
        s = verify.summary()
        assert s["total"] == n
        assert s["verdicts"] == {"asserted": 8000, "unknown": 4000}
        assert s["re_calibrate_rate"] == pytest.approx(1 / 3)
