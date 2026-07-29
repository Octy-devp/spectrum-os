"""Tests for anti-pollution double-layer quarantine — v2.5."""

import pytest
from spectrum_os.quantum.quarantine import (
    mechanical_filter, inspector_gate, quarantine_check, TEMPORAL_BLACKLIST
)

class TestMechanicalFilter:
    def test_clean_text(self):
        ok, matches = mechanical_filter('Elastic defense with railway concentration')
        assert ok is True
        assert matches == []

    def test_contaminated(self):
        ok, matches = mechanical_filter('blitzkrieg tactics with nuclear weapons')
        assert ok is False
        assert 'blitzkrieg' in matches or 'nuclear' in matches

    def test_case_insensitive(self):
        ok, matches = mechanical_filter('BLITZKRIEG through the Ardennes')
        assert ok is False

    def test_partial_match_no_false_positive(self):
        # "nation" should NOT match "nato"
        ok, matches = mechanical_filter('national defense strategy')
        assert ok is True

    def test_empty_text(self):
        ok, matches = mechanical_filter('')
        assert ok is True

class TestInspectorGate:
    def test_1915_clean(self):
        r = inspector_gate('Trench warfare in East Prussia', 1915)
        assert r['passed'] is True

    def test_1915_anachronism(self):
        r = inspector_gate('blitzkrieg tactics', 1915)
        assert r['passed'] is False

class TestQuarantineCheck:
    def test_pass(self):
        r = quarantine_check('Railway logistics for eastern front', 1915)
        assert r['passed'] is True

    def test_fail_mechanical(self):
        r = quarantine_check('blitzkrieg through the Ardennes', 1915)
        assert r['passed'] is False

    def test_blacklist_structure(self):
        assert isinstance(TEMPORAL_BLACKLIST, list)
        assert len(TEMPORAL_BLACKLIST) > 5
