"""Per-village monthly serialization for the GNP market suite (dynamics-gnp).

Engine-boundary finding: MarketDynamicsEngine.run_simulation already computes a
per-village monthly sold_share history internally (village_shares_history, one
37-point series m=0..36 per village) but previously only exported the final
value. These tests cover the retention/export hook:

- village_monthly_series(village_id, mode) accessor;
- build_village_monthly_document() -> schema draft s-trajectory-gnp-village-monthly-v1;
- endpoint consistency: monthly series last point == annual three-regime village
  values in village_transmission (tolerance 1e-6, same convention as the DKK
  S-trajectory v4 curation cross-check);
- full coverage: every village serialized, none dropped;
- backward compatibility: run_simulation's return dict keeps its exact key set
  (existing dynamics-gnp-{market}.json artifacts must not change shape).

Hermetic by default (synthetic villages + a pure config); one end-to-end test
runs against the real ECC village census when present, in-memory only (no files
written into the worldline repo).
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from spectrum_os.extensions.market_suite_dynamics import (  # noqa: E402
    MarketDynamicsEngine,
    VillageProfile,
    get_ar_frigorifico_config,
    get_all_market_configs,
    load_all_gnp_villages,
)

ECC_ROOT = "/home/octy/projects/ECC"

REGIMES = ("free_evolution", "cartel_lock", "cartel_split")


def _make_engine(n_villages: int = 8) -> MarketDynamicsEngine:
    cfg = get_ar_frigorifico_config()
    villages = [
        VillageProfile(
            village_id=f"synth-village-{i:03d}",
            zone_id=f"zone_{i % 3}",
            source_volume_1917=f"census/synth-village-{i:03d}-1917.json",
            source_volume_1914=f"census/synth-village-{i:03d}-1914.json",
            market_id="ar_frigorifico",
            crops=["wheat"],
            sold_share_1914=round(0.45 + 0.07 * (i % 8), 4),
        )
        for i in range(n_villages)
    ]
    return MarketDynamicsEngine(cfg, villages=villages)


def _run_suite(engine: MarketDynamicsEngine):
    suite = engine.run_comparative_suite()
    doc = engine.build_village_monthly_document(suite=suite)
    return suite, doc


def test_document_schema_shape():
    """Schema draft s-trajectory-gnp-village-monthly-v1: required fields, 37-point
    monthly grid 1914-01..1917-01, three regimes per village, values in [0.10, 1.0]."""
    engine = _make_engine(8)
    suite, doc = _run_suite(engine)

    assert doc["schema"] == "s-trajectory-gnp-village-monthly-v1"
    assert doc["track"] == "projection"
    assert doc["worldline"] == "beta_1_gnp"
    assert doc["market_id"] == "ar_frigorifico"
    assert doc["total_villages"] == 8
    assert doc["parameter_tagging"] == "[DERIVED_PROXY]"

    dates = doc["dates"]
    assert len(dates) == 37
    assert dates[0] == "1914-01"
    assert dates[-1] == "1917-01"
    # monthly contiguity: 1914-01..1914-12, 1915-01.., 1916-01.., 1917-01
    assert dates[11] == "1914-12" and dates[12] == "1915-01"
    assert dates[24] == "1916-01" and dates[25] == "1916-02"

    # GNP-side dimension declaration: market-structure quantity, NOT DKK S = R/(R+C)
    dim = doc["dimension_declaration"]
    assert "not_dkk_s" in dim and "R/(R+C)" in dim["not_dkk_s"]
    assert "sold_share" in dim["trajectory_quantity"]

    assert len(doc["villages"]) == 8
    ids = {v["village_id"] for v in doc["villages"]}
    assert ids == {f"synth-village-{i:03d}" for i in range(8)}

    for v in doc["villages"]:
        assert set(v["sold_share_monthly"].keys()) == set(REGIMES)
        for regime in REGIMES:
            series = v["sold_share_monthly"][regime]
            assert len(series) == 37
            for x in series:
                assert 0.10 <= x <= 1.0


def test_endpoint_consistency_with_annual_regime_values():
    """Monthly series last point == annual three-regime village values (<=1e-6),
    regime separation holds, and the cross-village mean matches the annual
    effective sold_share readout."""
    engine = _make_engine(8)
    suite, doc = _run_suite(engine)

    assert doc["endpoint_consistency"]["passed"] is True
    assert doc["endpoint_consistency"]["max_abs_deviation"] <= 1e-6
    assert doc["endpoint_consistency"]["cross_checked_against_suite"] is True

    suite_by_id = {vr["village_id"]: vr for vr in suite["village_transmission"]["villages"]}
    endpoint_field = {
        "free_evolution": "sold_share_1917_free",
        "cartel_lock": "sold_share_1917_lock",
        "cartel_split": "sold_share_1917_split",
    }
    for v in doc["villages"]:
        base = suite_by_id[v["village_id"]]
        assert base["sold_share_1914"] == v["sold_share_1914"]
        for regime, field in endpoint_field.items():
            series = v["sold_share_monthly"][regime]
            assert abs(series[-1] - base[field]) <= 1e-6

    # Regime separation: lock (kappa=1.0) and free evolution must not be identical
    v0 = doc["villages"][0]["sold_share_monthly"]
    assert any(a != b for a, b in zip(v0["free_evolution"], v0["cartel_lock"]))

    # Cross-village mean of final points == market-level annual effective sold_share
    for regime in REGIMES:
        mean_final = sum(
            v["sold_share_monthly"][regime][-1] for v in doc["villages"]
        ) / len(doc["villages"])
        annual = suite["regimes"][regime]["annual_readouts"]["1917"]["village_sold_share"]
        assert abs(round(mean_final, 4) - annual) <= 1e-3


def test_120_village_full_coverage():
    """With a 120-village panel every village is serialized exactly once, each with
    three complete 37-point series (no village dropped by the serializer)."""
    engine = _make_engine(120)
    suite, doc = _run_suite(engine)

    assert doc["total_villages"] == 120
    assert len(doc["villages"]) == 120
    ids = [v["village_id"] for v in doc["villages"]]
    assert len(set(ids)) == 120
    assert set(ids) == {f"synth-village-{i:03d}" for i in range(120)}
    for v in doc["villages"]:
        for regime in REGIMES:
            assert len(v["sold_share_monthly"][regime]) == 37


def test_accessor_and_backward_compat():
    """village_monthly_series accessor matches the document; unknown ids/modes raise
    KeyError; run_simulation's return dict keeps its exact legacy key set."""
    engine = _make_engine(5)

    # Accessor before any run must be loud, not silent-empty
    with pytest.raises(KeyError):
        engine.village_monthly_series("synth-village-000")

    # Document build before all three regimes ran must be loud
    engine.run_simulation(mode="free_evolution")
    with pytest.raises(ValueError):
        engine.build_village_monthly_document()

    suite = engine.run_comparative_suite()
    doc = engine.build_village_monthly_document(suite=suite)

    for v in doc["villages"]:
        for regime in REGIMES:
            assert engine.village_monthly_series(v["village_id"], regime) == \
                v["sold_share_monthly"][regime]
    # default mode is free_evolution
    assert engine.village_monthly_series("synth-village-000") == \
        doc["villages"][0]["sold_share_monthly"]["free_evolution"]

    with pytest.raises(KeyError):
        engine.village_monthly_series("no-such-village")
    with pytest.raises(KeyError):
        engine.village_monthly_series("synth-village-000", mode="no-such-mode")

    # Backward compatibility: the legacy return shape must be untouched
    res = engine.run_simulation(mode="free_evolution")
    assert set(res.keys()) == {
        "mode", "total_months", "annual_readouts", "monthly_series", "village_shares_final",
    }
    assert set(res["village_shares_final"].keys()) == {f"synth-village-{i:03d}" for i in range(5)}
    # no-village engines must refuse document building rather than emit an empty shell
    bare = MarketDynamicsEngine(get_ar_frigorifico_config(), villages=[])
    bare.run_comparative_suite()
    with pytest.raises(ValueError):
        bare.build_village_monthly_document()


def _needs_ecc():
    return Path(ECC_ROOT).exists()


def test_1917_window_parameterization():
    """Window parameterization: start_year=1917 yields the 1920-layer window
    1917-01..1920-01 (37 points). Terminal-year schema keys follow the window
    (sold_share_1920_*, feedback_metrics_1920, p_terminal_1920), annual readouts
    gain 1918/1919/1920, and endpoint consistency holds against the same window."""
    cfg = get_all_market_configs()['ar_frigorifico']
    # 1920 p_world canon must be wired (committed ECC build-1920-bifurcation-suite value)
    assert cfg.p_world_series[1920] == 110.0
    # 1918/1919 are exact linear thirds of the 1917→1920 ramp
    assert cfg.p_world_series[1918] == round(131.0 + (110.0 - 131.0) / 3.0, 4)
    assert cfg.p_world_series[1919] == round(131.0 + 2.0 * (110.0 - 131.0) / 3.0, 4)

    villages = [
        VillageProfile(
            village_id=f"synth-w17-{i:03d}",
            zone_id=f"zone_{i % 2}",
            source_volume_1917=f"census/synth-w17-{i:03d}-1917.json",
            source_volume_1914=f"census/synth-w17-{i:03d}-1914.json",
            market_id="ar_frigorifico",
            crops=["wheat"],
            sold_share_1914=0.80,
        )
        for i in range(6)
    ]
    engine = MarketDynamicsEngine(cfg, villages=villages)
    suite = engine.run_comparative_suite(start_year=1917)
    doc = engine.build_village_monthly_document(suite=suite, start_year=1917)

    # Monthly grid shifts to the 1917 window
    ms = suite['regimes']['free_evolution']['monthly_series']
    assert ms[0]['month_label'] == '1917-01' and ms[-1]['month_label'] == '1920-01'
    assert len(ms) == 37
    assert suite['regimes']['free_evolution']['annual_readouts']['1920']['month_label'] == '1920-01'
    assert '1918' in suite['regimes']['free_evolution']['annual_readouts']
    assert suite['time_horizon'] == '1905-1920'  # 1905 anchor pre-recorded, terminal 1920

    assert doc['dates'][0] == '1917-01' and doc['dates'][-1] == '1920-01'
    assert doc['dates'][12] == '1918-01'
    assert doc['time_horizon'] == '1917-01..1920-01'
    assert len(doc['dates']) == 37

    # Terminal-year key naming follows the window (no 1917-terminal keys left)
    vr = suite['village_transmission']['villages'][0]
    assert 'sold_share_1920_free' in vr and 'sold_share_1917_free' not in vr
    assert 'feedback_metrics_1920' in suite['village_transmission']
    assert 'p_terminal_1920' in suite['bifurcation_summary']

    # Endpoint consistency (monthly[-1] == annual 1920 regime value) must hold
    assert doc['endpoint_consistency']['passed'] is True
    assert doc['endpoint_consistency']['max_abs_deviation'] <= 1e-6
    suite_by_id = {v['village_id']: v for v in suite['village_transmission']['villages']}
    for v in doc['villages']:
        base = suite_by_id[v['village_id']]
        assert abs(v['sold_share_monthly']['free_evolution'][-1] - base['sold_share_1920_free']) <= 1e-6
        assert abs(v['sold_share_monthly']['cartel_lock'][-1] - base['sold_share_1920_lock']) <= 1e-6
        assert abs(v['sold_share_monthly']['cartel_split'][-1] - base['sold_share_1920_split']) <= 1e-6


def test_default_window_keeps_legacy_keys():
    """Backward compatibility: default start_year=1914 keeps every legacy terminal-
    year key name (sold_share_1917_*, feedback_metrics_1917, p_terminal_1917,
    time_horizon '1905-1917') so existing 1917-layer artifacts keep their shape."""
    engine = _make_engine(4)
    suite = engine.run_comparative_suite()

    vr = suite['village_transmission']['villages'][0]
    assert 'sold_share_1917_free' in vr and 'sold_share_1920_free' not in vr
    assert 'feedback_metrics_1917' in suite['village_transmission']
    assert 'p_terminal_1917' in suite['bifurcation_summary']
    assert suite['time_horizon'] == '1905-1917'

    doc = engine.build_village_monthly_document(suite=suite)
    assert doc['dates'][0] == '1914-01' and doc['dates'][-1] == '1917-01'
    assert doc['time_horizon'] == '1914-01..1917-01'
    assert doc['endpoint_consistency']['passed'] is True


@pytest.mark.skipif(not _needs_ecc(), reason="ECC worldline root not available")
def test_real_ecc_census_end_to_end_in_memory():
    """End-to-end against the real 120-village GNP census: all 7 markets covered,
    120 villages total, endpoint consistency holds everywhere. In-memory only —
    this test must never write into the worldline repo."""
    root = Path(ECC_ROOT)
    all_market_villages = load_all_gnp_villages(root)
    total = sum(len(v) for v in all_market_villages.values())
    assert total == 120, f"expected 120 real villages, found {total}"

    configs = get_all_market_configs()
    seen_ids = []
    for mid, cfg in configs.items():
        v_list = all_market_villages.get(mid, [])
        engine = MarketDynamicsEngine(cfg, villages=v_list)
        suite = engine.run_comparative_suite()
        doc = engine.build_village_monthly_document(suite=suite)
        assert doc["endpoint_consistency"]["passed"] is True, mid
        assert doc["total_villages"] == len(v_list)
        seen_ids.extend(v["village_id"] for v in doc["villages"])

    assert len(seen_ids) == 120
    assert len(set(seen_ids)) == 120
