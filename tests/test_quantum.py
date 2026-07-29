"""Tests for spectrum_os.quantum — role-vector tomography & decoherence watch.

Synthetic fixtures are hand-built multigraphs with known entropy values;
real-file tests parse the actual ECC data sources (skipped if absent).
"""

import math
import os

import numpy as np
import pytest

from spectrum_os.kernel._types import Verdict
from spectrum_os.quantum import (
    ROLES,
    RoleMultigraph,
    decoherence_watch,
    kl_divergence,
    load_knowledge_edges,
    load_warfare_tree,
    role_entropy,
    tomography,
)

LN2, LN3, LN4 = math.log(2), math.log(3), math.log(4)

ECC_KNOWLEDGE_EDGES = "/home/octy/projects/ECC/index/data/knowledge-dca-edges.yaml"
ECC_WARFARE_FIELDS = "/home/octy/projects/ECC/index/warfare/data/fields"

needs_ecc = pytest.mark.skipif(
    not os.path.exists(ECC_KNOWLEDGE_EDGES),
    reason="ECC data sources not available",
)


# ---------------------------------------------------------------------------
# Synthetic multigraph: exactly 2 threads (edges), 4 nodes — hand-computable
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_graph():
    """4 nodes {S, A, B, C}, 2 edges:

    * S --[alternative->direction]--> A   (S plays alternative, A plays direction)
    * B --[direction->crisis]--> S        (B plays direction, S plays crisis)

    S therefore superposes {alternative, crisis}; C is isolated.
    """
    graph = RoleMultigraph()
    graph.add_edge("S", "A", "alternative->direction", weight=0.9)
    graph.add_edge("B", "S", "direction->crisis", weight=0.9)
    return graph


def test_tomography_shared_state_superposition(tiny_graph):
    scan = tomography(tiny_graph, "S")
    assert scan.verdict is Verdict.ASSERTED
    assert scan.values["n_threads"] == 2
    dist = scan.values["role_distribution"]
    # equal thread weighting: one vote per thread -> 50/50
    assert dist["alternative"] == pytest.approx(0.5)
    assert dist["crisis"] == pytest.approx(0.5)
    assert dist["direction"] == 0.0
    assert dist["lag"] == 0.0
    assert scan.values["entropy"] == pytest.approx(LN2)
    assert scan.values["entropy_norm"] == pytest.approx(0.5)
    # PLAN-23 §6.3 Styx shadow: Alternative alive while Direction is zero
    assert scan.values["styx_shadow"] is True


def test_tomography_per_thread_role_vector(tiny_graph):
    scan = tomography(tiny_graph, "S")
    per_thread = scan.values["per_thread"]
    assert len(per_thread) == 2
    roles_seen = {next(iter(t)) for t in per_thread.values()}
    assert roles_seen == {"alternative", "crisis"}


def test_tomography_single_thread_contested(tiny_graph):
    scan = tomography(tiny_graph, "A")
    assert scan.verdict is Verdict.CONTESTED
    assert scan.values["n_threads"] == 1
    assert scan.values["role_distribution"]["direction"] == pytest.approx(1.0)
    assert scan.values["entropy"] == pytest.approx(0.0)


def test_tomography_absent_state_unknown(tiny_graph):
    scan = tomography(tiny_graph, "C-not-in-graph")
    assert scan.verdict is Verdict.UNKNOWN
    assert scan.values["n_threads"] == 0


def test_tomography_rejects_bad_weight_mode(tiny_graph):
    with pytest.raises(ValueError):
        tomography(tiny_graph, "S", weight_mode="vibes")


def test_add_edge_validates_relation():
    graph = RoleMultigraph()
    with pytest.raises(ValueError):
        graph.add_edge("X", "Y", "not-a-relation")
    with pytest.raises(ValueError):
        graph.add_edge("X", "Y", "foo->bar")


def test_kl_divergence_hand_computed():
    p = {"crisis": 0.5, "alternative": 0.5, "lag": 0.0, "direction": 0.0}
    uniform = {r: 0.25 for r in ROLES}
    assert kl_divergence(p, uniform) == pytest.approx(LN2)
    assert kl_divergence(uniform, uniform) == pytest.approx(0.0)


def test_role_entropy_values():
    assert role_entropy({r: 0.25 for r in ROLES}) == (pytest.approx(LN4), pytest.approx(1.0))
    assert role_entropy({"crisis": 1.0}) == (pytest.approx(0.0), pytest.approx(0.0))
    assert role_entropy(dict.fromkeys(ROLES, 0.0)) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Synthetic warfare time series — hand-built entropy cliff
# ---------------------------------------------------------------------------

def _quad(crisis="", lag="", alternative="", direction=""):
    """Minimal warfare branch entry (4-key schema)."""
    dca = {}
    if crisis:
        dca["crisis"] = crisis
    if lag:
        dca["lag"] = lag
    if alternative:
        dca["alternative"] = alternative
    if direction:
        dca["direction"] = direction
    return dca


@pytest.fixture
def cliff_tree():
    """Faction 'fx' over 4 rounds: full quad -> 3 roles -> 1 role -> 1 role.

    H: ln4 -> ln3 -> 0 -> 0. The r1->r2 drop (ln3) exceeds
    max(0.1, 0.5 * running_peak) = 0.5 * ln4 — exactly one alarm at r2.
    """
    data = {
        "factions": {
            "fx": [
                {"round": 0, "dca": _quad(crisis="c" * 40, lag="l" * 40,
                                          alternative="a" * 40, direction="d" * 40)},
                {"round": 1, "dca": _quad(crisis="c" * 40, lag="l" * 40,
                                          direction="d" * 40)},
                {"round": 2, "dca": _quad(direction="d" * 40)},
                {"round": 3, "dca": _quad(direction="d" * 40)},
            ]
        }
    }
    return RoleMultigraph.from_warfare_tree(data, system_id="synth")


def test_decoherence_entropy_cliff(cliff_tree):
    watch = decoherence_watch(cliff_tree, "fx")
    assert watch.verdict is Verdict.ASSERTED
    assert watch.values["n_points"] == 4
    entropies = [p["entropy"] for p in watch.values["series"]]
    assert entropies == pytest.approx([LN4, LN3, 0.0, 0.0])
    assert watch.values["alarm_detected"] is True
    assert len(watch.values["alarms"]) == 1
    alarm = watch.values["alarms"][0]
    assert alarm["t"] == 2.0
    assert alarm["drop"] == pytest.approx(LN3)


def test_decoherence_dominant_flips(cliff_tree):
    watch = decoherence_watch(cliff_tree, "fx")
    flips = watch.values["dominant_flips"]
    # r0/r1 dominant is 'crisis' (tie broken to first ROLES entry); r2 collapses to 'direction'
    assert any(f["from_role"] == "crisis" and f["to_role"] == "direction" for f in flips)
    assert all(not f["rule1_flip"] for f in flips)


def test_decoherence_rule1_flip():
    """direction -> crisis dominant flip = DCA rule #1 (§6.3)."""
    data = {
        "factions": {
            "fx": [
                {"round": 0, "dca": _quad(direction="d" * 50)},
                {"round": 1, "dca": _quad(crisis="c" * 50)},
            ]
        }
    }
    graph = RoleMultigraph.from_warfare_tree(data)
    watch = decoherence_watch(graph, "fx")
    assert watch.verdict is Verdict.CONTESTED  # only 2 points — thin series
    assert watch.values["dominant_flips"] == [
        {"t": 1.0, "from_role": "direction", "to_role": "crisis", "rule1_flip": True}
    ]


def test_decoherence_unknown_without_time_axis(tiny_graph):
    """Knowledge-style graph: no state_times -> honest UNKNOWN, no fabricated series."""
    watch = decoherence_watch(tiny_graph, "S")
    assert watch.verdict is Verdict.UNKNOWN
    assert watch.values["series"] == []


def test_placeholders_are_stale_not_active():
    data = {
        "factions": {
            "fx": [
                {"round": 0, "dca": {
                    "crisis": "(crisis empty)",
                    "lag": "延續上回合 Lag",
                    "alternative": " ",
                    "direction": "實質的方向內容",
                }},
            ]
        }
    }
    graph = RoleMultigraph.from_warfare_tree(data)
    scan = tomography(graph, "fx/r00")
    assert scan.values["role_distribution"]["direction"] == pytest.approx(1.0)
    assert scan.values["role_distribution"]["alternative"] == 0.0
    assert len(scan.values["stale_threads"]) == 1  # the three stale fields share one thread


def test_branch_aggregation_equal_thread_weighting():
    """Two branch entries in one round = two votes, averaged per-thread."""
    data = {
        "factions": {
            "fx": [
                {"round": 0, "dca": _quad(crisis="c" * 10, direction="d" * 10)},
                {"round": 0, "dca": _quad(direction="d" * 10)},
                {"round": 1, "dca": _quad(crisis="c" * 10, direction="d" * 10)},
            ]
        }
    }
    graph = RoleMultigraph.from_warfare_tree(data)
    scan = tomography(graph, "fx/r00")
    assert scan.values["n_threads"] == 2
    dist = scan.values["role_distribution"]
    # branch b0: (crisis .5, direction .5); branch b1: (direction 1.0) -> mean
    assert dist["crisis"] == pytest.approx(0.25)
    assert dist["direction"] == pytest.approx(0.75)


def test_mass_weight_mode_differs_from_presence():
    data = {
        "factions": {
            "fx": [
                {"round": 0, "dca": _quad(crisis="c" * 10, direction="d" * 90)},
            ]
        }
    }
    graph = RoleMultigraph.from_warfare_tree(data)
    presence = tomography(graph, "fx/r00", weight_mode="presence")
    mass = tomography(graph, "fx/r00", weight_mode="mass")
    assert presence.values["role_distribution"]["crisis"] == pytest.approx(0.5)
    assert mass.values["role_distribution"]["crisis"] == pytest.approx(0.1)


def test_warfare_key_mapping_chosen_path_and_strategic_intent():
    """5-key schema: chosen_path -> alternative, strategic_intent -> direction."""
    data = {
        "factions": {
            "fx": [
                {"round": 0, "dca": {
                    "crisis": "c", "lag": "l",
                    "chosen_path": "we choose this branch",
                    "strategic_intent": "our crystallized aim",
                    "doctrines": "not a DCA role — ignored",
                }},
            ]
        }
    }
    graph = RoleMultigraph.from_warfare_tree(data)
    scan = tomography(graph, "fx/r00")
    dist = scan.values["role_distribution"]
    for role in ROLES:
        assert dist[role] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Real ECC data sources (read-only)
# ---------------------------------------------------------------------------

@needs_ecc
def test_real_knowledge_multigraph_loads():
    graph = load_knowledge_edges(ECC_KNOWLEDGE_EDGES)
    assert len(graph.states()) == 204
    total_assignments = sum(len(v) for v in graph.assignments.values())
    assert total_assignments == 2 * 350  # one role per endpoint per edge


@needs_ecc
def test_real_knowledge_tomography_spielrein():
    """A genuinely superposed node: Alternative 6 threads, Direction 3 threads."""
    graph = load_knowledge_edges(ECC_KNOWLEDGE_EDGES)
    scan = tomography(graph, "230-spielrein-destruction-instinct")
    assert scan.verdict is Verdict.ASSERTED
    assert scan.values["n_threads"] == 9
    dist = scan.values["role_distribution"]
    assert dist["alternative"] == pytest.approx(6 / 9)
    assert dist["direction"] == pytest.approx(3 / 9)
    assert scan.values["entropy"] > 0.5
    assert scan.values["styx_shadow"] is False  # direction is nonzero


@needs_ecc
def test_real_knowledge_tomography_unknown_id():
    graph = load_knowledge_edges(ECC_KNOWLEDGE_EDGES)
    scan = tomography(graph, "999-no-such-entry")
    assert scan.verdict is Verdict.UNKNOWN


@needs_ecc
@pytest.mark.parametrize("field", [
    "experiment-track0",
    "anti-intervention-war-1915",
    "anti-intervention-war-cycle2",
])
def test_real_warfare_trees_parse(field):
    path = f"{ECC_WARFARE_FIELDS}/{field}/dca-branch-tree.yaml"
    if not os.path.exists(path):
        pytest.skip(f"{path} missing")
    graph = load_warfare_tree(path)
    factions = {sid.split("/")[0] for sid in graph.states()}
    assert factions == {"rus", "dsr"}
    assert len(graph.state_times) == len(graph.states())


@needs_ecc
def test_real_warfare_tomography_track0():
    graph = load_warfare_tree(f"{ECC_WARFARE_FIELDS}/experiment-track0/dca-branch-tree.yaml")
    scan = tomography(graph, "rus/r01")
    assert scan.verdict is Verdict.ASSERTED
    dist = scan.values["role_distribution"]
    for role in ROLES:
        assert dist[role] > 0.0  # round 1 quads fully populated


@needs_ecc
def test_real_warfare_decoherence_1915_rus():
    """The 1915 rus series shows an entropy window at rounds 5–7.

    PROVENANCE CAVEAT (20260728 rupture report): this window mixes genuine
    crystallization with measurement artifacts — template switch 5key→4key at
    r04→r05 (now flagged in ``state_meta[...]["template_switch"]``), carried-over
    stale fields, and alt==dir verbatim duplicates (now deduped with
    ``note="dup_text"``). Whether it is a real phase transition is API-gate
    interpretation work, not a mechanical given. The label "genuine entropy
    window" was premature.
    """
    graph = load_warfare_tree(
        f"{ECC_WARFARE_FIELDS}/anti-intervention-war-1915/dca-branch-tree.yaml")
    watch = decoherence_watch(graph, "rus")
    assert watch.verdict is Verdict.ASSERTED
    assert watch.values["n_points"] == 12  # r00 (all-stale setup) .. r11
    assert watch.values["alarm_detected"] is True
    first = watch.values["alarms"][0]
    assert first["t"] in (5.0, 6.0, 7.0)
    assert first["drop"] > 0.3
    ent_by_t = {p["t"]: p["entropy"] for p in watch.values["series"]}
    assert ent_by_t[1.0] > 1.0            # high superposition early
    assert ent_by_t[5.0] < ent_by_t[1.0]  # collapsed window


@needs_ecc
def test_real_warfare_decoherence_track0_no_false_alarm():
    """track0 rises from a sparse setup round into full quads — no drops."""
    graph = load_warfare_tree(f"{ECC_WARFARE_FIELDS}/experiment-track0/dca-branch-tree.yaml")
    watch = decoherence_watch(graph, "rus")
    assert watch.verdict is Verdict.ASSERTED
    assert watch.values["alarm_detected"] is False


@needs_ecc
def test_real_warfare_cycle2_thin_series_contested():
    graph = load_warfare_tree(
        f"{ECC_WARFARE_FIELDS}/anti-intervention-war-cycle2/dca-branch-tree.yaml")
    watch = decoherence_watch(graph, "rus")
    assert watch.verdict is Verdict.CONTESTED  # 2 rounds only


# ---------------------------------------------------------------------------
# §6.5 artifact filtering: dup_text dedup + template_switch markers
# ---------------------------------------------------------------------------

def test_dup_text_dedup_same_entry():
    """Verbatim-identical alternative/direction text = one measurement, not two."""
    data = {"factions": {"rus": [
        {"round": 1, "dca": {"crisis": "кризис", "lag": "лага",
                              "alternative": "SAME TEXT", "direction": "SAME TEXT"}},
    ]}}
    graph = RoleMultigraph.from_warfare_tree(data, "test")
    assigns = graph.state_assignments("rus/r01")
    active = {a.role: a for a in assigns if a.weight > 0}
    dupes = [a for a in assigns if a.note.startswith("dup_text:")]
    assert len(active) == 3  # crisis, lag, alternative(first) — direction deduped
    assert len(dupes) == 1 and dupes[0].role == "direction"
    assert dupes[0].weight == 0.0 and not dupes[0].stale


def test_dup_text_different_text_kept():
    data = {"factions": {"rus": [
        {"round": 1, "dca": {"alternative": "TEXT A", "direction": "TEXT B"}},
    ]}}
    graph = RoleMultigraph.from_warfare_tree(data, "test")
    active = [a for a in graph.state_assignments("rus/r01") if a.weight > 0]
    assert len(active) == 2
    assert all(not a.note for a in active)


def test_template_switch_marked():
    data = {"factions": {"rus": [
        {"round": 1, "dca": {"crisis": "x", "chosen_path": "y", "strategic_intent": "z"}},
        {"round": 2, "dca": {"crisis": "x", "alternative": "y", "direction": "z"}},
    ]}}
    graph = RoleMultigraph.from_warfare_tree(data, "test")
    assert graph.state_meta["rus/r01"]["template"] == "5key"
    assert "template_switch" not in graph.state_meta["rus/r01"]
    assert graph.state_meta["rus/r02"]["template"] == "4key"
    assert graph.state_meta["rus/r02"]["template_switch"] is True
