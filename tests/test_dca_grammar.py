"""Tests for DCA grammar gate — v2.5."""

import pytest
from spectrum_os.quantum.dca_grammar import (
    DCA_GRAMMAR, validate_alternative, validate_alternative_chain, mutate_alternative
)

class TestValidateAlternative:
    def test_valid_transitions(self):
        assert validate_alternative('crisis', 'lag') is True
        assert validate_alternative('crisis', 'alternative') is True
        assert validate_alternative('crisis', 'crisis') is True  # can deepen
        assert validate_alternative('lag', 'crisis') is True
        assert validate_alternative('alternative', 'direction') is True
        assert validate_alternative('direction', 'crisis') is True

    def test_forbidden(self):
        assert validate_alternative('direction', 'alternative') is False
        assert validate_alternative('lag', 'direction') is False

    def test_depth_gate(self):
        # At recursion depth 6, only 'direction' is allowed
        assert validate_alternative('alternative', 'alternative', depth=6) is False
        assert validate_alternative('alternative', 'direction', depth=6) is True

    def test_unknown_role(self):
        assert validate_alternative('crisis', 'unknown_role') is False


class TestValidateChain:
    def test_full_cycle(self):
        ok, _ = validate_alternative_chain(['crisis', 'lag', 'alternative', 'direction'])
        assert ok is True

    def test_forbidden_chain(self):
        ok, reason = validate_alternative_chain(['crisis', 'lag', 'direction'])
        assert ok is False
        assert 'lag' in reason and 'direction' in reason

    def test_empty_chain(self):
        ok, _ = validate_alternative_chain([])
        assert ok is True

    def test_single_role(self):
        ok, _ = validate_alternative_chain(['crisis'])
        assert ok is True


class TestMutate:
    def test_from_crisis(self):
        muts = mutate_alternative('crisis')
        assert 'lag' in muts
        assert 'alternative' in muts
        assert 'direction' not in muts  # crisis cannot directly become direction

    def test_from_direction(self):
        muts = mutate_alternative('direction')
        assert 'crisis' in muts
        assert 'lag' in muts
        assert 'direction' in muts  # can persist
        assert 'alternative' not in muts  # forbidden

    def test_at_depth_limit(self):
        muts = mutate_alternative('alternative', depth=6)
        assert muts == ['direction']  # only direction allowed at depth limit
