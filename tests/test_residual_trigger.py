"""Tests for synth/residual_trigger.py (N4: Mode-B residual trigger, S7-5)."""

import json

from spectrum_os.synth.residual_trigger import (
    load_residual_table,
    make_residual_trigger,
    should_enumerate,
)


class TestShouldEnumerate:
    def test_int_keyed_hit(self):
        assert should_enumerate({5: {"criteria": ["saturation"]}}, 5) is True

    def test_int_keyed_miss(self):
        assert should_enumerate({5: {"criteria": ["saturation"]}}, 3) is False

    def test_json_round_trip_string_key(self):
        # JSON serialises int keys to strings; an int ``t`` must still hit.
        assert should_enumerate({"5": {"criteria": ["saturation"]}}, 5) is True

    def test_date_keyed_hit(self):
        table = {"1914-08-02": {"criteria": ["saturation"]}}
        assert should_enumerate(table, "1914-08-02") is True

    def test_month_keyed_hit(self):
        table = {"1916-05": {"criteria": ["decoupling"]}}
        assert should_enumerate(table, "1916-05") is True

    def test_any_sharp_criterion_hits(self):
        assert should_enumerate({7: {"criteria": ["saturation"]}}, 7) is True

    def test_gap_alone_does_not_trigger(self):
        # Gap is amplifier-only: a pure-gap entry never fires by itself —
        # otherwise arc-granularity gap coverage (100% of the timeline) would
        # degenerate Mode B from "spectrum failure" into "whole timeline".
        assert should_enumerate({7: {"criteria": ["gap"]}}, 7) is False

    def test_correlation_flip_alone_does_not_trigger(self):
        # correlation_flip is engine-demoted (v2.10, N6): its 91-month net
        # (1913-06..1920-12) covers nearly the whole query window — the same
        # wide-net degeneration as gap. A pure-correlation_flip entry never
        # fires, on day keys or month keys.
        assert (
            should_enumerate({7: {"criteria": ["correlation_flip"]}}, 7) is False
        )
        assert (
            should_enumerate(
                {"1916-05": {"criteria": ["correlation_flip"]}}, "1916-05"
            )
            is False
        )

    def test_gap_with_sharp_fires_via_sharp(self):
        # gap riding alongside a sharp criterion fires — but only because the
        # sharp signal is present (gap amplifies, never ignites).
        assert should_enumerate({7: {"criteria": ["gap", "saturation"]}}, 7) is True

    def test_correlation_flip_with_sharp_fires_via_sharp(self):
        # correlation_flip riding alongside a sharp criterion fires — but only
        # because the sharp signal is present (amplifies, never ignites).
        assert (
            should_enumerate(
                {"1916-05": {"criteria": ["correlation_flip", "decoupling"]}},
                "1916-05",
            )
            is True
        )
        assert (
            should_enumerate(
                {"1916-05": {"criteria": ["correlation_flip", "saturation"]}},
                "1916-05",
            )
            is True
        )

    def test_gap_with_correlation_flip_does_not_fire(self):
        # Both are amplifier-only — together they still never ignite: no sharp
        # criterion ①–② is present to ride on.
        assert (
            should_enumerate(
                {"1916-05": {"criteria": ["gap", "correlation_flip"]}}, "1916-05"
            )
            is False
        )

    def test_day_key_miss_falls_back_to_month_key(self):
        # 1914-08-02 is absent but the month key 1914-08 is present — the
        # day-key miss must narrow to the month prefix and hit decoupling.
        table = {"1914-08": {"criteria": ["decoupling"]}}
        assert should_enumerate(table, "1914-08-02") is True

    def test_day_key_miss_falls_back_to_month_key_gap_is_still_false(self):
        # The month-prefix fallback reaches month-keyed gap entries too — but
        # pure gap never fires, so the day-key miss stays False.
        table = {"1914-08": {"criteria": ["gap"]}}
        assert should_enumerate(table, "1914-08-02") is False

    def test_day_key_miss_falls_back_to_month_key_correlation_flip_is_still_false(
        self,
    ):
        # The month-prefix fallback reaches month-keyed correlation_flip too —
        # but pure correlation_flip never fires (engine-demoted), so the
        # day-key miss stays False.
        table = {"1914-08": {"criteria": ["correlation_flip"]}}
        assert should_enumerate(table, "1914-08-02") is False

    def test_day_key_hit_precedes_month_fallback(self):
        # A direct day-key hit still fires before the month fallback.
        table = {
            "1914-08-02": {"criteria": ["saturation"]},
            "1914-08": {"criteria": ["correlation_flip"]},
        }
        assert should_enumerate(table, "1914-08-02") is True

    def test_day_key_miss_without_month_key(self):
        assert (
            should_enumerate(
                {"1915-01": {"criteria": ["saturation"]}}, "1915-02-10"
            )
            is False
        )

    def test_no_table(self):
        assert should_enumerate(None, 0) is False

    def test_empty_table(self):
        assert should_enumerate({}, 0) is False

    def test_empty_criteria_is_no_hit(self):
        assert should_enumerate({7: {"criteria": []}}, 7) is False

    def test_missing_criteria_field_is_no_hit(self):
        assert should_enumerate({7: {"detail": {"x": 1}}}, 7) is False

    def test_unknown_criterion_is_no_hit(self):
        assert should_enumerate({7: {"criteria": ["not_a_criterion"]}}, 7) is False

    def test_sensitivity_above_threshold(self):
        cb = lambda t, context=None: 0.9  # noqa: E731
        assert should_enumerate({}, 0, sensitivity_cb=cb) is True

    def test_sensitivity_at_threshold(self):
        cb = lambda t, context=None: 0.5  # noqa: E731
        assert (
            should_enumerate({}, 0, sensitivity_cb=cb, sensitivity_threshold=0.5)
            is True
        )

    def test_sensitivity_below_threshold(self):
        cb = lambda t, context=None: 0.3  # noqa: E731
        assert should_enumerate({}, 0, sensitivity_cb=cb) is False

    def test_sensitivity_without_table(self):
        cb = lambda t, context=None: 0.8  # noqa: E731
        assert should_enumerate(None, 0, sensitivity_cb=cb) is True

    def test_sensitivity_receives_context(self):
        seen = {}

        def cb(t, context=None):
            seen["t"] = t
            seen["ctx"] = context
            return 1.0

        should_enumerate({}, 42, sensitivity_cb=cb, context={"date": "1914-08-02"})
        assert seen == {"t": 42, "ctx": {"date": "1914-08-02"}}

    def test_table_hit_takes_precedence_over_low_cb(self):
        cb = lambda t, context=None: 0.0  # noqa: E731
        assert (
            should_enumerate({3: {"criteria": ["saturation"]}}, 3, sensitivity_cb=cb)
            is True
        )


class TestLoadResidualTable:
    def test_from_dict(self):
        table = load_residual_table({5: {"criteria": ["saturation"]}})
        assert should_enumerate(table, 5) is True

    def test_from_json_path(self, tmp_path):
        p = tmp_path / "residuals.json"
        p.write_text(
            json.dumps({"1914-08-02": {"criteria": ["saturation"]}}), encoding="utf-8"
        )
        table = load_residual_table(str(p))
        assert should_enumerate(table, "1914-08-02") is True

    def test_wrapped_generator_format(self, tmp_path):
        p = tmp_path / "residuals.json"
        p.write_text(
            json.dumps(
                {"_meta": {"v": 1}, "residuals": {5: {"criteria": ["decoupling"]}}}
            ),
            encoding="utf-8",
        )
        table = load_residual_table(str(p))
        assert "_meta" not in table
        assert should_enumerate(table, 5) is True

    def test_missing_file_safe(self):
        assert load_residual_table("/nonexistent/table.json") == {}

    def test_broken_json_safe(self, tmp_path):
        p = tmp_path / "broken.json"
        p.write_text("{not json", encoding="utf-8")
        assert load_residual_table(str(p)) == {}

    def test_none_safe(self):
        assert load_residual_table(None) == {}

    def test_non_mapping_non_path_source_safe(self):
        # A non-str/Mapping source (e.g. int) must degrade to {} — never raise
        # a TypeError (defect-3 regression guard).
        assert load_residual_table(123) == {}
        assert load_residual_table(3.14) == {}


class TestMakeResidualTrigger:
    def test_identity_key(self):
        trigger = make_residual_trigger({9: {"criteria": ["saturation"]}})
        assert trigger(9, {}) is True
        assert trigger(8, {}) is False

    def test_key_fn_maps_to_date(self):
        trigger = make_residual_trigger(
            {"1914-08-02": {"criteria": ["saturation"]}},
            key_fn=lambda t, ctx: ctx["date"],
        )
        assert trigger(3, {"date": "1914-08-02"}) is True
        assert trigger(3, {"date": "1914-01-01"}) is False

    def test_none_table(self):
        trigger = make_residual_trigger(None)
        assert trigger(0, None) is False

    def test_sensitivity_wired_through(self):
        trigger = make_residual_trigger(None, sensitivity_cb=lambda t, context=None: 0.9)
        assert trigger(0, {}) is True
