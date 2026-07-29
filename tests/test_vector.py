"""Tests for 6D state vector — v2.5 vector contract."""

import pytest
import math
import random
from spectrum_os.quantum.vector import (
    StateVector, project_ternary, vector_from_roles, norm, to_array
)

class TestStateVector:
    def test_creation_defaults(self):
        v = StateVector()
        assert v.d1 == 0 and v.d4 == 0.0

    def test_to_dict_roundtrip(self):
        v = StateVector(d1=1, d2=-1, d3=0, d4=3.5, d5=1.57, d6=0.8)
        d = v.to_dict()
        v2 = StateVector.from_dict(d)
        assert v == v2

    def test_norm(self):
        v = StateVector(d1=1, d2=-1, d3=0, d4=3.5, d5=1.57, d6=0.8)
        assert v.norm() > 0
        assert abs(v.norm() - norm(v)) < 0.001

    def test_to_array(self):
        v = StateVector(d1=1, d2=-1, d3=0, d4=3.5, d5=1.57, d6=0.8)
        arr = to_array(v)
        assert arr == [1.0, -1.0, 0.0, 3.5, 1.57, 0.8]

    def test_random_roundtrip(self):
        for _ in range(100):
            v = StateVector(
                d1=random.choice([-1, 0, 1]),
                d2=random.choice([-1, 0, 1]),
                d3=random.choice([-1, 0, 1]),
                d4=random.uniform(0, 100),
                d5=random.uniform(-math.pi, math.pi),
                d6=random.uniform(0, 1),
            )
            assert StateVector.from_dict(v.to_dict()) == v

class TestProjectTernary:
    def test_direction_dominant(self):
        r = project_ternary({'direction': 0.9, 'crisis': 0.1, 'lag': 0.0, 'alternative': 0.0})
        assert r == (1, -1, -1)

    def test_crisis_dominant(self):
        r = project_ternary({'direction': 0.0, 'crisis': 0.9, 'lag': 0.1, 'alternative': 0.0})
        assert r == (-1, -1, -1)

    def test_balanced(self):
        r = project_ternary({'direction': 0.25, 'crisis': 0.25, 'lag': 0.25, 'alternative': 0.25})
        # d1: 0, d2: 0 (lag=0.25 = threshold), d3: 0 (alt=0.25 = mean of others)
        assert r == (0, 0, 0)

    def test_empty_dict(self):
        r = project_ternary({})
        assert r == (0, 0, 0)

    def test_partial_keys(self):
        r = project_ternary({'direction': 0.5})
        assert r[0] == 1  # direction dominant, d1=+1

class TestVectorFromRoles:
    def test_with_spectrum(self):
        v = vector_from_roles(
            {'direction': 0.4, 'crisis': 0.3, 'lag': 0.2, 'alternative': 0.1},
            {'freq': 3.5, 'phase': 1.57, 'amplitude': 0.8}
        )
        assert v.d4 == 3.5
        assert v.d5 == 1.57
        assert v.d6 == 0.8
        assert isinstance(v.d1, int)  # ternary direction is discrete

    def test_without_spectrum(self):
        v = vector_from_roles({'direction': 0.5, 'crisis': 0.5})
        assert v.d4 == 0.0 and v.d5 == 0.0 and v.d6 == 0.0

    def test_ternary_not_caged(self):
        """Ban check: D4-D6 must not be constrained by ternary."""
        v = vector_from_roles(
            {'direction': 0.4, 'crisis': 0.3, 'lag': 0.2, 'alternative': 0.1},
            {'freq': 99.0, 'phase': 6.28, 'amplitude': 0.99}  # far beyond any ternary range
        )
        assert v.d4 == 99.0   # NOT clamped to {-1,0,1}
        assert v.d6 == 0.99   # NOT clamped
