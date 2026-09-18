"""Spectrum OS Multi-Market Oligopoly Dynamics Kernel (dynamics-gnp)

Universal N-Actor Oligopoly Simulation Engine implementing:
- SLOT-MIRROR-SCHEMA.md §六: GNP World (Great National Powers, beta_1).
- Lenin (1916) thesis on monopoly and competition coexistence.
- Peasant withholding / hoarding capacity idling feedback (delta_idle * (1 - sold_share_effective) * kappa).
- Three regimes: Free Evolution (endogenous kappa(t)), Cartel Lock (kappa=1.0), Cartel Split (kappa=0.05).
- 120 GNP village empirical transmission grounded in 360 observation-gnp volumes.
- All derived parameters tagged with [DERIVED_PROXY].
- Window parameterization: start_year=1914 (legacy 1914-01..1917-01, default;
  byte-identical outputs) or start_year=1917 (1920 observation layer,
  1917-01..1920-01, p_world 1920 canon per P_WORLD_1920_CANON).
"""

from __future__ import annotations

import math
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class VillageProfile:
    village_id: str
    zone_id: str
    source_volume_1917: str
    source_volume_1914: str
    market_id: str
    crops: List[str]
    sold_share_1914: float  # Empirical baseline from 1914 volume census


@dataclass
class MarketConfig:
    market_id: str
    commodity: str
    geography: str
    base_unit: str
    source_volumes: List[str]
    actors: Dict[str, Dict[str, Any]]
    
    # Macro price cycle (1913=100)
    p_world_series: Dict[int, float]  # e.g. {1905: 90, 1914: 95, 1915: 120, 1916: 152, 1917: 123}
    base_terminal_price: float
    base_farmgate_price_1914: float
    freight_and_fees: float
    monopsony_markdown_max: float
    monopsony_markdown_comp: float
    
    # Dynamic parameters [DERIVED_PROXY]
    kappa_0: float
    alpha_kappa: float
    beta_kappa: float
    sigma_p: float
    gamma_s: float
    
    # Rural elasticity transmission
    village_sold_share_0: float
    elasticity_p: float
    elasticity_kappa: float
    delta_idle: float = 0.20  # [DERIVED_PROXY]: Trust capacity idling penalty parameter
    
    # Historical anchor values (1905, 1914, 1917)
    historical_anchors: Dict[int, Dict[str, float]] = field(default_factory=dict)
    lenin_thesis_notes: Dict[str, str] = field(default_factory=dict)


def classify_village_market(rel_path_str: str, data: Dict[str, Any]) -> str:
    """Classify an observation volume into one of the 7 empirical commodity markets."""
    p = rel_path_str.lower()
    z = str(data.get('zone_id', '')).lower()

    # 1. Argentine frigorifico & River Plate agro-export
    if any(k in p for k in ['argentina', 'bahia-blanca', 'rosario']) or z.startswith('ar_'):
        return 'ar_frigorifico'

    # 2. US Cotton South debt crop lien
    if any(k in p for k in ['alabama', 'georgia-cotton', 'memphis', 'new-orleans', 'vicksburg']) or any(
        z.startswith(k) for k in ['us_alabama', 'us_georgia', 'us_tennessee', 'us_louisiana', 'us_mississippi']
    ):
        return 'us_furnish_lien'

    # 3. PRD silk filature & commercial agriculture
    if 'prd-' in p or z.startswith('cn_prd'):
        return 'prd_silk_filature'

    # 4. Japan rice tenancy & zaibatsu
    if any(k in p for k in ['tohoku', 'kanto', 'kyushu', 'shinano']) or z.startswith('jp_'):
        return 'jp_rice_tenancy'

    # 5. India / Ceylon / Malaya Chettiar credit & imperial export
    if any(k in p for k in ['benares', 'bengal', 'ceylon', 'deccan', 'madras', 'malaya', 'punjab']) or any(
        z.startswith(k) for k in ['in_', 'lk_', 'my_']
    ):
        return 'in_chettiar_credit'

    # 6. UK Corn factor ring
    if p.startswith('uk/') or z.startswith('uk_'):
        return 'uk_corn_factor'

    # 7. US Midwest / North American grain & continental
    return 'us_midwest_grain'


def _require_root(root_path: Optional[Path]) -> Path:
    """Resolve the world-data root.

    This layer holds a *domain model*, not world data: the kernel stays
    world-agnostic (docs/PLAN.md) and so must every path here. Callers supply
    their own root (an ECC caller passes its repo root; another worldline would
    pass its own directory tree) — a machine-specific default would silently
    bind the library to one world.
    """
    if root_path is None:
        raise ValueError(
            "root_path is required: this module reads world data and must not "
            "assume a hard-coded worldline directory"
        )
    return Path(root_path)


def load_all_gnp_villages(root_path: Optional[Path] = None) -> Dict[str, List[VillageProfile]]:
    """Load and classify all 120 GNP villages, extracting empirical 1914 sold_share baselines."""
    root_path = _require_root(root_path)
    base_dir = root_path / 'index/locations/data/almanac/1910-1920/1917/observation-gnp'
    
    files_17 = sorted(list(base_dir.glob('*/*-1917.json')))
    files_17 = [f for f in files_17 if 'monitor' not in f.name and 'registry' not in f.name]
    
    market_villages: Dict[str, List[VillageProfile]] = {}
    
    for f17 in files_17:
        stem = f17.name.replace('-1917.json', '')
        f14 = f17.parent / f'{stem}-1914.json'
        
        with open(f17, 'r', encoding='utf-8') as fp:
            d17 = json.load(fp)
        
        with open(f14, 'r', encoding='utf-8') as fp:
            d14 = json.load(fp)
            
        rel_path = str(f17.relative_to(base_dir))
        m_id = classify_village_market(rel_path, d17)
        
        # Extract empirical sold_share from 1914 census
        hh_shares = []
        for hh in d14.get('census', []):
            ss = hh.get('household_year', {}).get('market_pulse', {}).get('sold_share')
            if ss is not None:
                hh_shares.append(float(ss))
        avg_ss_1914 = sum(hh_shares) / len(hh_shares) if hh_shares else 0.80
        
        crops = []
        if d17.get('census'):
            crops = d17['census'][0].get('household_year', {}).get('production', {}).get('crops', [])
            
        vp = VillageProfile(
            village_id=stem,
            zone_id=d17.get('zone_id', stem),
            source_volume_1917=str(f17.relative_to(root_path)),
            source_volume_1914=str(f14.relative_to(root_path)),
            market_id=m_id,
            crops=crops,
            sold_share_1914=round(avg_ss_1914, 4),
        )
        market_villages.setdefault(m_id, []).append(vp)
        
    return market_villages


class MarketDynamicsEngine:
    def __init__(self, config: MarketConfig, villages: Optional[List[VillageProfile]] = None):
        self.cfg = config
        self.villages = villages or []
        # Per-village monthly sold_share history, stashed by run_simulation:
        # {mode: {village_id: [float x (total_months+1)]}}. This is a retention of
        # state the loop already computes (village_shares_history) — no physics change.
        self.village_monthly: Dict[str, Dict[str, List[float]]] = {}

    def _interpolate_p_world(self, month_idx: int, start_year: int = 1914) -> float:
        """Piecewise-linear monthly p_world over the window [start_year .. start_year+3].

        Default start_year=1914 reproduces the legacy 1914-01..1917-01 window
        exactly (same anchors, same segment arithmetic). The 1917 start needs
        anchors 1917..1920 (see extend_p_world_to_1920).
        """
        series = self.cfg.p_world_series
        p0 = series[start_year]
        p1 = series[start_year + 1]
        p2 = series[start_year + 2]
        p3 = series[start_year + 3]

        if month_idx <= 12:
            return p0 + (p1 - p0) * (month_idx / 12.0)
        elif month_idx <= 24:
            return p1 + (p2 - p1) * ((month_idx - 12) / 12.0)
        else:
            return p2 + (p3 - p2) * ((month_idx - 24) / 12.0)

    def run_simulation(
        self,
        mode: str = 'free_evolution',
        delta_idle_override: Optional[float] = None,
        start_year: int = 1914,
    ) -> Dict[str, Any]:
        actor_ids = list(self.cfg.actors.keys())
        a1_id, a2_id = actor_ids[0], actor_ids[1]
        a1_data, a2_data = self.cfg.actors[a1_id], self.cfg.actors[a2_id]
        
        c1 = a1_data.get('unit_processing_cost', a1_data.get('unit_processing_cost_gbp_tonne', 1.0))
        c2 = a2_data.get('unit_processing_cost', a2_data.get('unit_processing_cost_gbp_tonne', 1.0))
        delta_c = abs(c1 - c2)
        
        s1 = a1_data.get('baseline_share', a1_data.get('quota_baseline_share', 0.60))
        s2 = a2_data.get('baseline_share', a2_data.get('quota_baseline_share', 0.40))
        
        delta_idle = self.cfg.delta_idle if delta_idle_override is None else delta_idle_override
        
        if mode == 'cartel_lock':
            kappa = 1.0
        elif mode == 'cartel_split':
            kappa = self.cfg.kappa_0
        else:
            kappa = self.cfg.kappa_0

        monthly_records = []
        annual_readouts = {}

        # Pre-record 1905 anchor
        if 1905 in self.cfg.historical_anchors:
            anch05 = self.cfg.historical_anchors[1905]
            annual_readouts['1905'] = anch05

        total_months = 36
        dt = 1.0 / 12.0

        # Village state tracking
        village_shares_history: Dict[str, List[float]] = {v.village_id: [] for v in self.villages}

        for m in range(total_months + 1):
            year = start_year + m // 12
            month_in_year = (m % 12) + 1
            month_label = f'{year}-{month_in_year:02d}'

            p_w_idx = self._interpolate_p_world(m, start_year)
            p_terminal = self.cfg.base_terminal_price * (p_w_idx / 100.0)

            # dP_world / dt approximation
            if m < total_months:
                dp_dt = (self._interpolate_p_world(m + 1, start_year) - p_w_idx) / dt
            else:
                dp_dt = (p_w_idx - self._interpolate_p_world(m - 1, start_year)) / dt

            # Mode overrides for kappa
            if mode == 'cartel_lock':
                kappa = 1.0
            elif mode == 'cartel_split':
                if m >= 24:  # Beginning of 1916-1917 bust
                    decay = math.exp(-0.35 * (m - 24))
                    kappa = 0.05 + (self.cfg.kappa_0 - 0.05) * decay
                else:
                    kappa = min(1.0, self.cfg.kappa_0 + 0.003 * m)

            # Pricing & Markdown
            markdown = self.cfg.monopsony_markdown_comp + kappa * (
                self.cfg.monopsony_markdown_max - self.cfg.monopsony_markdown_comp
            )
            raw_farmgate = p_terminal * (1.0 - markdown) - self.cfg.freight_and_fees
            p_farmgate = max(0.1, round(raw_farmgate, 2))
            retention_rate = round(p_farmgate / p_terminal, 4)

            # Profit margins
            fob_net = p_terminal - self.cfg.freight_and_fees
            margin_1 = round(fob_net - p_farmgate - c1, 2)
            margin_2 = round(fob_net - p_farmgate - c2, 2)

            # Replicator dynamics for market share
            avg_margin = s1 * margin_1 + s2 * margin_2
            if mode != 'cartel_lock':
                ds1_dt = self.cfg.gamma_s * (1.0 - kappa) * s1 * (margin_1 - avg_margin)
                s1 = max(0.05, min(0.95, s1 + ds1_dt * dt))
                s2 = 1.0 - s1
            hhi = round(s1 ** 2 + s2 ** 2, 4)

            # Village-level sold_share transmission & aggregated effective sold_share
            p_ratio = p_farmgate / max(0.01, self.cfg.base_farmgate_price_1914)
            current_village_shares = []
            
            if self.villages:
                for v in self.villages:
                    # Elastic response formula
                    v_raw = v.sold_share_1914 * (p_ratio ** self.cfg.elasticity_p) * (
                        1.0 - self.cfg.elasticity_kappa * (kappa - self.cfg.kappa_0)
                    )
                    v_ss = min(1.0, max(0.10, round(v_raw, 4)))
                    current_village_shares.append(v_ss)
                    village_shares_history[v.village_id].append(v_ss)
                effective_sold_share = round(sum(current_village_shares) / len(current_village_shares), 4)
            else:
                sold_share_raw = self.cfg.village_sold_share_0 * (p_ratio ** self.cfg.elasticity_p) * (
                    1.0 - self.cfg.elasticity_kappa * (kappa - self.cfg.kappa_0)
                )
                effective_sold_share = min(1.0, max(0.10, round(sold_share_raw, 4)))

            # Capacity idling penalty
            penalty_idle = round(delta_idle * (1.0 - effective_sold_share) * kappa, 4)

            # Free Evolution ODE step with closed-loop delta_idle feedback
            if mode == 'free_evolution' and m < total_months:
                incentive_cooperate = self.cfg.alpha_kappa * math.tanh(dp_dt / self.cfg.sigma_p) * (1.0 - kappa)
                incentive_cheat = self.cfg.beta_kappa * (1.0 if dp_dt < -1.0 else 0.0) * kappa * (delta_c + 0.1)
                dkappa_dt = incentive_cooperate - incentive_cheat - penalty_idle
                kappa = max(0.05, min(1.0, kappa + dkappa_dt * dt))

            record = {
                'month_idx': m,
                'month_label': month_label,
                'year': year,
                'p_world_index': round(p_w_idx, 2),
                'p_terminal': round(p_terminal, 2),
                'p_farmgate': p_farmgate,
                'retention_rate': retention_rate,
                'kappa': round(kappa, 4),
                'share_actor_1': round(s1, 4),
                'share_actor_2': round(s2, 4),
                'hhi': hhi,
                'margin_actor_1': margin_1,
                'margin_actor_2': margin_2,
                'village_sold_share': effective_sold_share,
                'penalty_idle': penalty_idle,
            }
            # Add aliases for ar_frigorifico backwards compatibility
            if self.cfg.market_id == 'ar_frigorifico':
                record['share_fg01'] = round(s1, 4)
                record['share_fg02'] = round(s2, 4)
                record['p_terminal_gbp'] = round(p_terminal, 2)
                record['p_farmgate_gbp'] = p_farmgate

            monthly_records.append(record)

            if month_in_year == 1 and str(year) not in annual_readouts:
                annual_readouts[str(year)] = record

        # Retain the full per-village monthly series (37 points, m=0..36) computed
        # inside the loop. Values are the engine's own monthly sold_share outputs —
        # not interpolation. The return dict stays unchanged so existing
        # dynamics-gnp-{market}.json artifacts keep their exact shape.
        if self.villages:
            self.village_monthly[mode] = {
                v.village_id: list(village_shares_history[v.village_id]) for v in self.villages
            }

        return {
            'mode': mode,
            'total_months': total_months,
            'annual_readouts': annual_readouts,
            'monthly_series': monthly_records,
            'village_shares_final': {v.village_id: village_shares_history[v.village_id][-1] for v in self.villages}
            if self.villages else {},
        }

    def village_monthly_series(self, village_id: str, mode: str = 'free_evolution') -> List[float]:
        """Return the per-village monthly sold_share series for a regime.

        Series length is total_months + 1 (37 points over the engine window,
        dates[0] = {start_year}-01 .. dates[36] = {start_year+3}-01), matching
        the market-level monthly_series grid. Raises KeyError if the mode has not
        been run or the village_id is unknown.
        """
        if mode not in self.village_monthly:
            raise KeyError(
                f"no village monthly history for mode '{mode}'; "
                "run_simulation(mode=...) must run first"
            )
        series = self.village_monthly[mode]
        if village_id not in series:
            raise KeyError(
                f"unknown village_id '{village_id}' for market '{self.cfg.market_id}'"
            )
        return list(series[village_id])

    def build_village_monthly_document(
        self,
        suite: Optional[Dict[str, Any]] = None,
        endpoint_tolerance: float = 1e-6,
        start_year: int = 1914,
    ) -> Dict[str, Any]:
        """Build the S-trajectory-gnp per-village monthly document (draft schema
        ``s-trajectory-gnp-village-monthly-v1``).

        Requires that all three regimes have been run on this engine (i.e. after
        run_comparative_suite) **with the same ``start_year``**. If ``suite`` (the
        run_comparative_suite output) is supplied, monthly series endpoints are
        cross-checked against the annual three-regime village values in
        village_transmission (tolerance ``endpoint_tolerance``, default 1e-6 —
        same convention as the DKK S-trajectory v4 curation cross-check).
        """
        terminal_year = start_year + 3
        required_modes = ('free_evolution', 'cartel_lock', 'cartel_split')
        missing = [m for m in required_modes if m not in self.village_monthly]
        if missing:
            raise ValueError(
                f"village monthly history missing for regimes {missing}; "
                "run_comparative_suite() must run first"
            )
        if not self.villages:
            raise ValueError(
                "no villages loaded for market "
                f"'{self.cfg.market_id}'; per-village monthly serialization is undefined"
            )

        dates = [f'{start_year + m // 12}-{(m % 12) + 1:02d}' for m in range(36 + 1)]

        village_entries = []
        max_abs_dev = 0.0
        suite_villages = {}
        if suite is not None:
            suite_villages = {
                vr['village_id']: vr for vr in suite['village_transmission']['villages']
            }
        endpoint_field = {
            'free_evolution': f'sold_share_{terminal_year}_free',
            'cartel_lock': f'sold_share_{terminal_year}_lock',
            'cartel_split': f'sold_share_{terminal_year}_split',
        }

        for v in self.villages:
            series_block = {}
            for mode in required_modes:
                s = self.village_monthly[mode][v.village_id]
                series_block[mode] = s
                if suite is not None and v.village_id in suite_villages:
                    expected = suite_villages[v.village_id][endpoint_field[mode]]
                    max_abs_dev = max(max_abs_dev, abs(s[-1] - expected))
            village_entries.append({
                'village_id': v.village_id,
                'zone_id': v.zone_id,
                'market_id': self.cfg.market_id,
                'source_volume_1917': v.source_volume_1917,
                'sold_share_1914': v.sold_share_1914,
                'sold_share_monthly': series_block,
            })

        endpoint_consistency = {
            'check': 'sold_share_monthly[regime][-1] == village_transmission sold_share_1917_{free,lock,split}',
            'tolerance': endpoint_tolerance,
            'max_abs_deviation': round(max_abs_dev, 10),
            'cross_checked_against_suite': suite is not None,
            'passed': max_abs_dev <= endpoint_tolerance,
        }

        return {
            'schema': 's-trajectory-gnp-village-monthly-v1',
            'track': 'projection',
            'worldline': 'beta_1_gnp',
            'market_id': self.cfg.market_id,
            'commodity': self.cfg.commodity,
            'geography': self.cfg.geography,
            'base_unit': self.cfg.base_unit,
            'dimension_declaration': {
                'trajectory_quantity': 'village sold_share (marketed share of harvest, clamped [0.10, 1.0])',
                'not_dkk_s': 'This is NOT the DKK ForceField S = R/(R+C). Per SLOT-MIRROR-SCHEMA §六, '
                             'the GNP-side dynamics dimension is a market-structure quantity '
                             '(p_farmgate / retention_rate / kappa / hhi / village_sold_share); '
                             'kappa(t) here is the collusion index in [0,1], not a force ratio.',
                'kappa_range': '[0, 1]',
            },
            'time_horizon': f'{start_year}-01..{terminal_year}-01',
            'dates': dates,
            'total_villages': len(self.villages),
            'model_specification': 'sold_share_v(t) = sold_share_v,1914 * (P_farmgate(t) / P_1914)^epsilon_p * [1 - epsilon_kappa * (kappa(t) - kappa_0)]',
            'parameter_tagging': '[DERIVED_PROXY]',
            'provenance': {
                'canon_ruling': 'SINGLE_CANON_DIRECT_ADJUSTMENT',
                'framework': 'SLOT-MIRROR-SCHEMA §六 + Lenin 1916 Monopoly-Competition Coexistence + Closed-Loop Feedback',
                'source_volumes': self.cfg.source_volumes,
                'generator': 'spectrum_os.extensions.market_suite_dynamics.MarketDynamicsEngine.build_village_monthly_document',
            },
            'villages': village_entries,
            'endpoint_consistency': endpoint_consistency,
        }

    def run_comparative_suite(
        self,
        delta_idle_override: Optional[float] = None,
        start_year: int = 1914,
    ) -> Dict[str, Any]:
        terminal_year = start_year + 3
        res_free = self.run_simulation(
            mode='free_evolution', delta_idle_override=delta_idle_override, start_year=start_year
        )
        res_split = self.run_simulation(
            mode='cartel_split', delta_idle_override=delta_idle_override, start_year=start_year
        )
        res_lock = self.run_simulation(
            mode='cartel_lock', delta_idle_override=delta_idle_override, start_year=start_year
        )

        t_free = res_free['annual_readouts'][str(terminal_year)]
        t_split = res_split['annual_readouts'][str(terminal_year)]
        t_lock = res_lock['annual_readouts'][str(terminal_year)]

        delta_idle_val = self.cfg.delta_idle if delta_idle_override is None else delta_idle_override

        # Build village transmission detailed block
        village_records = []
        if self.villages:
            for v in self.villages:
                ss_free = res_free['village_shares_final'].get(v.village_id, v.sold_share_1914)
                ss_split = res_split['village_shares_final'].get(v.village_id, v.sold_share_1914)
                ss_lock = res_lock['village_shares_final'].get(v.village_id, v.sold_share_1914)
                village_records.append({
                    'village_id': v.village_id,
                    'zone_id': v.zone_id,
                    'source_volume_1917': v.source_volume_1917,
                    'sold_share_1914': v.sold_share_1914,
                    f'sold_share_{terminal_year}_free': ss_free,
                    f'sold_share_{terminal_year}_lock': ss_lock,
                    f'sold_share_{terminal_year}_split': ss_split,
                    'delta_sold_share_split_vs_lock': round(ss_split - ss_lock, 4),
                })

        mean_1914 = round(sum(v.sold_share_1914 for v in self.villages) / len(self.villages), 4) if self.villages else self.cfg.village_sold_share_0
        mean_t_free = t_free['village_sold_share']
        mean_t_lock = t_lock['village_sold_share']
        mean_t_split = t_split['village_sold_share']

        village_transmission = {
            'model_specification': 'sold_share_v(t) = sold_share_v,1914 * (P_farmgate(t) / P_1914)^epsilon_p * [1 - epsilon_kappa * (kappa(t) - kappa_0)]',
            'feedback_ode_specification': 'dkappa/dt = incentive_cooperate - incentive_cheat - delta_idle * (1.0 - mean(sold_share_v)) * kappa',
            'delta_idle': delta_idle_val,
            'parameter_tagging': '[DERIVED_PROXY]',
            'total_villages': len(self.villages),
            'village_panel_summary': {
                'mean_sold_share_1914': mean_1914,
                f'mean_sold_share_{terminal_year}_free': mean_t_free,
                f'mean_sold_share_{terminal_year}_lock': mean_t_lock,
                f'mean_sold_share_{terminal_year}_split': mean_t_split,
                'delta_split_vs_lock': round(mean_t_split - mean_t_lock, 4),
            },
            'villages': village_records,
            'effective_sold_share_trajectory': {
                'free_evolution': {
                    yr: ro['village_sold_share'] for yr, ro in res_free['annual_readouts'].items()
                },
                'cartel_lock': {
                    yr: ro['village_sold_share'] for yr, ro in res_lock['annual_readouts'].items()
                },
                'cartel_split': {
                    yr: ro['village_sold_share'] for yr, ro in res_split['annual_readouts'].items()
                },
            },
            f'feedback_metrics_{terminal_year}': {
                'effective_sold_share_free': mean_t_free,
                'effective_sold_share_lock': mean_t_lock,
                'effective_sold_share_split': mean_t_split,
                'idle_capacity_penalty_free': t_free['penalty_idle'],
                'idle_capacity_penalty_lock': t_lock['penalty_idle'],
                'idle_capacity_penalty_split': t_split['penalty_idle'],
            },
        }

        bifurcation = {
            f'p_terminal_{terminal_year}': t_free['p_terminal'],
            'kappa_comparison': {
                'free_evolution': t_free['kappa'],
                'cartel_split': t_split['kappa'],
                'cartel_lock': t_lock['kappa'],
                'delta_split_vs_lock': round(t_split['kappa'] - t_lock['kappa'], 4),
            },
            f'market_shares_actor_2_{terminal_year}': {
                'free_evolution': t_free['share_actor_2'],
                'cartel_split': t_split['share_actor_2'],
                'cartel_lock': t_lock['share_actor_2'],
                'actor_2_gain_under_split': round(t_split['share_actor_2'] - t_lock['share_actor_2'], 4),
            },
            f'village_farmgate_price_{terminal_year}': {
                'free_evolution': t_free['p_farmgate'],
                'cartel_split': t_split['p_farmgate'],
                'cartel_lock': t_lock['p_farmgate'],
                'farmgate_premium_split_vs_lock': round(t_split['p_farmgate'] - t_lock['p_farmgate'], 2),
            },
            f'village_sold_share_{terminal_year}': {
                'free_evolution': mean_t_free,
                'cartel_split': mean_t_split,
                'cartel_lock': mean_t_lock,
                'bifurcation_delta_sold_share': round(mean_t_split - mean_t_lock, 4),
            },
            f'oligarch_total_margin_{terminal_year}': {
                'free_evolution': round(t_free['margin_actor_1'] + t_free['margin_actor_2'], 2),
                'cartel_split': round(t_split['margin_actor_1'] + t_split['margin_actor_2'], 2),
                'cartel_lock': round(t_lock['margin_actor_1'] + t_lock['margin_actor_2'], 2),
            },
            'lenin_proposition_readout': self.cfg.lenin_thesis_notes,
        }

        schema_name = 'dynamics-gnp-demonstration-market' if self.cfg.market_id == 'ar_frigorifico' else 'dynamics-gnp-market-simulation'

        return {
            'schema': schema_name,
            'market_id': self.cfg.market_id,
            'commodity': self.cfg.commodity,
            'geography': self.cfg.geography,
            'base_unit': self.cfg.base_unit,
            'time_horizon': f'1905-{terminal_year}' if 1905 in self.cfg.historical_anchors else f'{start_year}-{terminal_year}',
            'provenance': {
                'canon_ruling': 'SINGLE_CANON_DIRECT_ADJUSTMENT',
                'framework': 'SLOT-MIRROR-SCHEMA §六 + Lenin 1916 Monopoly-Competition Coexistence + Closed-Loop Feedback',
                'source_volumes': self.cfg.source_volumes,
                'parameter_tagging': '[DERIVED_PROXY]',
            },
            'actors': self.cfg.actors,
            'regimes': {
                'free_evolution': res_free,
                'cartel_split': res_split,
                'cartel_lock': res_lock,
            },
            'village_transmission': village_transmission,
            'bifurcation_summary': bifurcation,
        }


# =========================================================================
# MARKET DEFINITIONS (ALL 7 MARKETS)
# =========================================================================

def get_ar_frigorifico_config() -> MarketConfig:
    return MarketConfig(
        market_id='ar_frigorifico',
        commodity='chilled_beef_carcass',
        geography='argentina_buenos_aires',
        base_unit='gbp_per_carcass_tonne',
        source_volumes=[
            'index/locations/data/almanac/1910-1920/1917/observation-gnp/sphere/argentina-beef-frigorifico-periurban-1914.json',
            'index/locations/data/almanac/1910-1920/1917/observation-gnp/sphere/argentina-beef-frigorifico-periurban-1917.json',
            'index/locations/data/almanac/1910-1920/1917/observation-gnp/sphere/argentina-beef-frigorifico-secondary-1914.json',
            'index/locations/data/almanac/1910-1920/1917/observation-gnp/sphere/argentina-beef-frigorifico-secondary-1917.json',
            'index/locations/data/almanac/1910-1920/1917/observation-gnp/sphere/argentina-wheat-buenosaires-primary-1917.json',
        ],
        actors={
            'FG_01': {
                'actor_id': 'FG_01',
                'name': 'FG_01 英資冷藏聯合公司 (Anglo-South American Meat Cartel)',
                'capital_origin': 'british_imperial',
                'capacity_slaughter_head_day': 750,
                'annual_slaughter_head_1914': 157500,
                'annual_slaughter_head_1917': 150000,
                'carcass_tonne_1914': 31900.0,
                'carcass_tonne_1917': 30300.0,
                'unit_processing_cost_gbp_tonne': 4.2,
                'quota_baseline_share': 0.68,
                'orientation': 'quota_defense',
            },
            'FG_02': {
                'actor_id': 'FG_02',
                'name': 'FG_02 北美資本新廠 (Swift & Armour River Plate Syndicate)',
                'capital_origin': 'us_sphere',
                'capacity_slaughter_head_day': 400,
                'annual_slaughter_head_1914': 74000,
                'annual_slaughter_head_1917': 72000,
                'carcass_tonne_1914': 14900.0,
                'carcass_tonne_1917': 14500.0,
                'unit_processing_cost_gbp_tonne': 3.8,
                'quota_baseline_share': 0.32,
                'orientation': 'quota_challenger',
            },
        },
        p_world_series={1905: 85.0, 1914: 95.0, 1915: 118.0, 1916: 154.0, 1917: 131.0},
        base_terminal_price=42.0,
        base_farmgate_price_1914=22.0,
        freight_and_fees=6.5,
        monopsony_markdown_max=0.32,
        monopsony_markdown_comp=0.12,
        kappa_0=0.65,
        alpha_kappa=0.18,
        beta_kappa=0.28,
        sigma_p=8.5,
        gamma_s=0.12,
        village_sold_share_0=0.913,
        elasticity_p=0.08,
        elasticity_kappa=0.12,
        delta_idle=0.20,
        historical_anchors={
            1905: {
                'year': 1905,
                'p_world_index': 85.0,
                'p_terminal': 35.70,
                'p_farmgate': 18.56,
                'retention_rate': 0.5199,
                'kappa': 0.58,
                'share_actor_1': 0.70,
                'share_actor_2': 0.30,
                'hhi': 0.5800,
                'village_sold_share': 0.88,
            }
        },
        lenin_thesis_notes={
            'thesis': 'Monopoly does not eliminate competition, but exists above and alongside it (Lenin 1916).',
            'mechanism': 'High price cycle (1915-16) cements cartel collusion; price down-draught (1916-17) ignites quota friction and packing capacity idling struggles.',
            'village_impact': 'Under Cartel-Lock, monopsony markdown extracts £10.85/tonne into trust coffers, depressing ranchgate sold_share to 0.894; under Cartel-Split, packing rivalry raises farmgate price to £38.12/t (+36.9%) and lifts sold_share to 1.000.',
        },
    )


def get_us_midwest_grain_config() -> MarketConfig:
    return MarketConfig(
        market_id='us_midwest_grain',
        commodity='corn_and_wheat',
        geography='us_midwest_corn_wheat_belt',
        base_unit='gbp_tonne_and_cents_per_bushel',
        source_volumes=[
            'index/locations/data/almanac/1910-1920/1917/observation-gnp/usa/illinois-corn-soy-chicago-primary-1917.json',
            'index/locations/data/almanac/1910-1920/1917/observation-gnp/usa/illinois-corn-soy-chicago-secondary-1914.json',
            'index/locations/data/almanac/1910-1920/1917/observation-gnp/usa/minnesota-springwheat-dairy-primary-1917.json',
            'index/locations/data/almanac/1910-1920/1917/observation-gnp/usa/kansas-wheat-chicago-primary-1905.json',
        ],
        actors={
            'cbot_grain_trust': {
                'actor_id': 'cbot_grain_trust',
                'name': 'CBOT 穀物升降機托拉斯與鐵路聯合 (Chicago Terminal Elevator Pool)',
                'capital_origin': 'us_sphere',
                'baseline_share': 0.72,
                'unit_processing_cost': 0.85,
                'orientation': 'line_elevator_monopoly',
            },
            'independent_coop_elevators': {
                'actor_id': 'independent_coop_elevators',
                'name': '農民合作升降機與獨立託運行 (Farmer Co-op Elevator Federation)',
                'capital_origin': 'us_sphere_independent',
                'baseline_share': 0.28,
                'unit_processing_cost': 0.95,
                'orientation': 'basis_bid_challenger',
            },
        },
        p_world_series={1905: 88.0, 1914: 95.0, 1915: 120.0, 1916: 152.0, 1917: 123.0},
        base_terminal_price=14.0,
        base_farmgate_price_1914=8.82,
        freight_and_fees=2.2,
        monopsony_markdown_max=0.28,
        monopsony_markdown_comp=0.10,
        kappa_0=0.54,
        alpha_kappa=0.16,
        beta_kappa=0.24,
        sigma_p=8.0,
        gamma_s=0.15,
        village_sold_share_0=0.96,
        elasticity_p=0.06,
        elasticity_kappa=0.10,
        delta_idle=0.20,
        historical_anchors={
            1905: {
                'year': 1905,
                'p_world_index': 88.0,
                'p_terminal': 12.32,
                'p_farmgate': 7.82,
                'retention_rate': 0.6347,
                'kappa': 0.48,
                'share_actor_1': 0.74,
                'share_actor_2': 0.26,
                'hhi': 0.6152,
                'village_sold_share': 0.95,
            }
        },
        lenin_thesis_notes={
            'thesis': 'Rail-elevator trust oligopoly extraction via basis markdown and car allocation.',
            'mechanism': '1915-16 wartime grain demand expands elevator margins, reinforcing pooling; 1917 European recovery drops price to 123 index, triggering co-op elevator basis bidding friction.',
            'village_impact': 'Under Cartel-Lock, local basis widens (-8 cents/bu), depressing farmer price to £7.68/t; under Cartel-Split, elevator bidding war narrows basis, lifting farmgate to £9.73/t (+26.7%) and sold_share to 0.99.',
        },
    )


def get_uk_corn_factor_config() -> MarketConfig:
    return MarketConfig(
        market_id='uk_corn_factor',
        commodity='barley_wheat_malt',
        geography='uk_east_anglia_home_counties',
        base_unit='gbp_tonne_and_shillings_quarter',
        source_volumes=[
            'index/locations/data/almanac/1910-1920/1917/observation-gnp/uk/east-anglia-suffolk-london-secondary-1914.json',
            'index/locations/data/almanac/1910-1920/1917/observation-gnp/uk/home-counties-kent-london-periurban-1905.json',
            'index/locations/data/almanac/1910-1920/1917/observation-gnp/uk/home-counties-kent-london-periurban-1917.json',
        ],
        actors={
            'london_mark_lane_factors': {
                'actor_id': 'london_mark_lane_factors',
                'name': '倫敦馬克巷糧商公會 (Mark Lane Corn Exchange Factor Ring)',
                'capital_origin': 'british_imperial',
                'baseline_share': 0.62,
                'unit_processing_cost': 0.70,
                'orientation': 'consignment_quota_defense',
            },
            'provincial_merchant_dealers': {
                'actor_id': 'provincial_merchant_dealers',
                'name': '地方麥芽與市鎮獨立糧商 (Provincial Corn Dealers & Maltsters)',
                'capital_origin': 'british_imperial_local',
                'baseline_share': 0.38,
                'unit_processing_cost': 0.82,
                'orientation': 'local_cash_purchase_bidder',
            },
        },
        p_world_series={1905: 85.0, 1914: 94.0, 1915: 116.0, 1916: 148.0, 1917: 126.0},
        base_terminal_price=16.5,
        base_farmgate_price_1914=11.22,
        freight_and_fees=1.8,
        monopsony_markdown_max=0.25,
        monopsony_markdown_comp=0.08,
        kappa_0=0.52,
        alpha_kappa=0.15,
        beta_kappa=0.25,
        sigma_p=8.0,
        gamma_s=0.14,
        village_sold_share_0=0.88,
        elasticity_p=0.05,
        elasticity_kappa=0.08,
        delta_idle=0.20,
        historical_anchors={
            1905: {
                'year': 1905,
                'p_world_index': 85.0,
                'p_terminal': 14.03,
                'p_farmgate': 9.48,
                'retention_rate': 0.6757,
                'kappa': 0.49,
                'share_actor_1': 0.64,
                'share_actor_2': 0.36,
                'hhi': 0.5392,
                'village_sold_share': 0.87,
            }
        },
        lenin_thesis_notes={
            'thesis': 'Metropolitan commission factors extracting consignment spreads over tenant maltsters.',
            'mechanism': 'Factors defend 0.49-0.58 collusion band through London Baltic/Mark Lane ring settlement; deflationary turn in 1917 encourages provincial direct buying.',
            'village_impact': 'Cartel-Lock depresses East Anglian farmer barley return to £10.12/t; Cartel-Split allows provincial cash bids at £12.78/t (+26.3%), restoring tenant farmer surplus.',
        },
    )


def get_in_chettiar_credit_config() -> MarketConfig:
    return MarketConfig(
        market_id='in_chettiar_credit',
        commodity='paddy_rice_and_jute',
        geography='british_india_bengal_madras',
        base_unit='rupee_per_maund_and_gbp_equivalent',
        source_volumes=[
            'index/locations/data/almanac/1910-1920/1917/observation-gnp/empire/bengal-jute-calcutta-primary-1917.json',
            'index/locations/data/almanac/1910-1920/1917/observation-gnp/empire/madras-cotton-madras-primary-1917.json',
            'index/locations/data/almanac/1910-1920/1917/observation-gnp/empire/benares-primary-1905.json',
        ],
        actors={
            'chettiar_moneylender_syndicate': {
                'actor_id': 'chettiar_moneylender_syndicate',
                'name': '納圖考泰·遮地也爾公會 (Nattukottai Chettiar Banking Guild)',
                'capital_origin': 'british_imperial_indigenous',
                'baseline_share': 0.68,
                'unit_processing_cost': 1.10,
                'orientation': 'compound_interest_crop_pledge',
            },
            'marwari_arhtiya_dealers': {
                'actor_id': 'marwari_arhtiya_dealers',
                'name': '馬爾瓦爾糧食代理與期貨牙商 (Marwari Arhtiya Dealer Network)',
                'capital_origin': 'british_imperial_marwari',
                'baseline_share': 0.32,
                'unit_processing_cost': 1.35,
                'orientation': 'mandi_spot_forward_arbitrage',
            },
        },
        p_world_series={1905: 82.0, 1914: 92.0, 1915: 110.0, 1916: 135.0, 1917: 118.0},
        base_terminal_price=22.0,
        base_farmgate_price_1914=10.56,
        freight_and_fees=3.2,
        monopsony_markdown_max=0.42,
        monopsony_markdown_comp=0.18,
        kappa_0=0.68,
        alpha_kappa=0.18,
        beta_kappa=0.22,
        sigma_p=7.5,
        gamma_s=0.10,
        village_sold_share_0=0.42,
        elasticity_p=0.08,
        elasticity_kappa=0.14,
        delta_idle=0.20,
        historical_anchors={
            1905: {
                'year': 1905,
                'p_world_index': 82.0,
                'p_terminal': 18.04,
                'p_farmgate': 8.74,
                'retention_rate': 0.4845,
                'kappa': 0.64,
                'share_actor_1': 0.70,
                'share_actor_2': 0.30,
                'hhi': 0.5800,
                'village_sold_share': 0.40,
            }
        },
        lenin_thesis_notes={
            'thesis': 'Indigenous usury-merchant capital merging into imperial export clearing chains.',
            'mechanism': 'Chettiar interest rate collusion (24-37%) combines with Marwari mandi purchasing pools to monopolize harvest delivery before paddy leaves the threshing floor.',
            'village_impact': 'Cartel-Lock forces ryots to deliver paddy at distressed £9.08/t equivalent, leaving marketed share at barely 0.395; Cartel-Split allows competitive mandi bids, lifting farmgate to £14.22/t (+56.6%).',
        },
    )


def get_us_furnish_lien_config() -> MarketConfig:
    return MarketConfig(
        market_id='us_furnish_lien',
        commodity='upland_raw_cotton',
        geography='us_cotton_belt_black_belt',
        base_unit='gbp_tonne_and_cents_per_pound',
        source_volumes=[
            'index/locations/data/almanac/1910-1920/1917/observation-gnp/usa/alabama-tenant-montgomery-periurban-1917.json',
            'index/locations/data/almanac/1910-1920/1917/observation-gnp/usa/georgia-cotton-savannah-primary-1917.json',
            'index/locations/data/almanac/1910-1920/1917/observation-gnp/usa/vicksburg-primary-1914.json',
        ],
        actors={
            'furnish_merchant_planter_ring': {
                'actor_id': 'furnish_merchant_planter_ring',
                'name': '棉南物資墊支雜貨商與種植園主公會 (Furnish Merchant & Planter Ring)',
                'capital_origin': 'us_sphere_south',
                'baseline_share': 0.74,
                'unit_processing_cost': 1.50,
                'orientation': 'crop_lien_store_debt_monopoly',
            },
            'independent_ginners_cotton_factors': {
                'actor_id': 'independent_ginners_cotton_factors',
                'name': '獨立軋花廠與港口棉花代理商 (Independent Ginners & Port Factors)',
                'capital_origin': 'us_sphere_merchant',
                'baseline_share': 0.26,
                'unit_processing_cost': 1.75,
                'orientation': 'cash_cotton_bidding_challenger',
            },
        },
        p_world_series={1905: 86.0, 1914: 90.0, 1915: 125.0, 1916: 165.0, 1917: 135.0},
        base_terminal_price=28.0,
        base_farmgate_price_1914=12.88,
        freight_and_fees=3.5,
        monopsony_markdown_max=0.48,
        monopsony_markdown_comp=0.15,
        kappa_0=0.76,
        alpha_kappa=0.14,
        beta_kappa=0.26,
        sigma_p=8.5,
        gamma_s=0.12,
        village_sold_share_0=0.68,
        elasticity_p=0.07,
        elasticity_kappa=0.15,
        delta_idle=0.20,
        historical_anchors={
            1905: {
                'year': 1905,
                'p_world_index': 86.0,
                'p_terminal': 24.08,
                'p_farmgate': 11.12,
                'retention_rate': 0.4618,
                'kappa': 0.72,
                'share_actor_1': 0.76,
                'share_actor_2': 0.24,
                'hhi': 0.6352,
                'village_sold_share': 0.66,
            }
        },
        lenin_thesis_notes={
            'thesis': 'Pre-capitalist debt bondage reinforced by monopoly merchant furnish credit.',
            'mechanism': 'Furnish merchants enforce 20-25% store markups and 25% nominal debt interest, holding 0.74-0.80 territorial concentration over tenant crop liens.',
            'village_impact': 'Cartel-Lock locks sharecroppers into negative balance sheets with £13.82/t net return; Cartel-Split allows cash ginner access, raising farmgate to £22.04/t (+59.5%) and liberating tenant retention.',
        },
    )


def get_prd_silk_filature_config() -> MarketConfig:
    return MarketConfig(
        market_id='prd_silk_filature',
        commodity='raw_mulberry_silk',
        geography='cn_pearl_river_delta_shunde',
        base_unit='tael_per_picul_and_gbp_equivalent',
        source_volumes=[
            'index/locations/data/almanac/1910-1920/1917/observation-gnp/empire/prd-shunde-mulberry-hongkong-primary-1917.json',
            'index/locations/data/almanac/1910-1920/1917/observation-gnp/empire/prd-dongguan-rice-hongkong-secondary-1914.json',
            'index/locations/data/almanac/1910-1920/1917/observation-gnp/empire/prd-kowloon-ntl-village-hongkong-periurban-1905.json',
        ],
        actors={
            'shunde_steam_filature_association': {
                'actor_id': 'shunde_steam_filature_association',
                'name': '順德容奇機器繅絲公會 (Shunde Steam Filature Guild)',
                'capital_origin': 'british_hongkong_canton_national',
                'baseline_share': 0.65,
                'unit_processing_cost': 2.40,
                'orientation': 'cocoon_procurement_price_cap',
            },
            'canton_silk_export_houses': {
                'actor_id': 'canton_silk_export_houses',
                'name': '沙面十三行與香港洋行買辦 (Shameen Foreign Silk Export Houses)',
                'capital_origin': 'british_imperial_mercantile',
                'baseline_share': 0.35,
                'unit_processing_cost': 2.80,
                'orientation': 'terminal_export_margin_capture',
            },
        },
        p_world_series={1905: 80.0, 1914: 92.0, 1915: 108.0, 1916: 142.0, 1917: 120.0},
        base_terminal_price=45.0,
        base_farmgate_price_1914=22.50,
        freight_and_fees=4.0,
        monopsony_markdown_max=0.38,
        monopsony_markdown_comp=0.14,
        kappa_0=0.64,
        alpha_kappa=0.17,
        beta_kappa=0.25,
        sigma_p=8.0,
        gamma_s=0.12,
        village_sold_share_0=0.78,
        elasticity_p=0.09,
        elasticity_kappa=0.12,
        delta_idle=0.20,
        historical_anchors={
            1905: {
                'year': 1905,
                'p_world_index': 80.0,
                'p_terminal': 36.00,
                'p_farmgate': 18.24,
                'retention_rate': 0.5067,
                'kappa': 0.60,
                'share_actor_1': 0.67,
                'share_actor_2': 0.33,
                'hhi': 0.5578,
                'village_sold_share': 0.76,
            }
        },
        lenin_thesis_notes={
            'thesis': 'Export processing compradors subordinating mulberry-dike peasant domestic economy.',
            'mechanism': 'Filature association convenes seasonal price conventions, fixing cocoon collection discount; export house advances enforce raw silk lien.',
            'village_impact': 'Cartel-Lock caps peasant cocoon price at £23.18/t equivalent, holding marketed surplus to 0.748; Cartel-Split fosters filature bidding competition, elevating farmgate to £33.42/t (+44.2%).',
        },
    )


def get_jp_rice_tenancy_config() -> MarketConfig:
    return MarketConfig(
        market_id='jp_rice_tenancy',
        commodity='brown_paddy_rice',
        geography='japan_kanto_tohoku_niigata',
        base_unit='yen_per_koku_and_gbp_equivalent',
        source_volumes=[
            'index/locations/data/almanac/1910-1920/1917/observation-gnp/sphere/tohoku-rice-tokyo-primary-1917.json',
            'index/locations/data/almanac/1910-1920/1917/observation-gnp/sphere/kanto-orchard-yokohama-primary-1914.json',
            'index/locations/data/almanac/1910-1920/1917/observation-gnp/sphere/shinano-sericulture-tokyo-secondary-1905.json',
        ],
        actors={
            'absentee_parasite_landlords': {
                'actor_id': 'absentee_parasite_landlords',
                'name': '不在村寄生地主會與豪農協議會 (Absentee Parasitic Landlord Guild)',
                'capital_origin': 'japan_sphere_landlord',
                'baseline_share': 0.60,
                'unit_processing_cost': 1.20,
                'orientation': 'in_kind_55pct_rent_enforcement',
            },
            'zaibatsu_rice_dealers': {
                'actor_id': 'zaibatsu_rice_dealers',
                'name': '三井三菱米穀問屋與青田買投機商 (Zaibatsu Rice Wholesalers & Aota-gai Dealers)',
                'capital_origin': 'japan_sphere_zaibatsu',
                'baseline_share': 0.40,
                'unit_processing_cost': 1.45,
                'orientation': 'military_procurement_arbitrage',
            },
        },
        p_world_series={1905: 90.0, 1914: 96.0, 1915: 105.0, 1916: 115.0, 1917: 122.0},
        base_terminal_price=18.0,
        base_farmgate_price_1914=8.64,
        freight_and_fees=1.6,
        monopsony_markdown_max=0.52,  # 50-55% in-kind parasitic rent
        monopsony_markdown_comp=0.22,
        kappa_0=0.72,
        alpha_kappa=0.16,
        beta_kappa=0.22,
        sigma_p=7.5,
        gamma_s=0.10,
        village_sold_share_0=0.52,
        elasticity_p=0.10,
        elasticity_kappa=0.16,
        delta_idle=0.20,
        historical_anchors={
            1905: {
                'year': 1905,
                'p_world_index': 90.0,
                'p_terminal': 16.20,
                'p_farmgate': 7.55,
                'retention_rate': 0.4660,
                'kappa': 0.68,
                'share_actor_1': 0.62,
                'share_actor_2': 0.38,
                'hhi': 0.5288,
                'village_sold_share': 0.48,
            }
        },
        lenin_thesis_notes={
            'thesis': 'Parasitic landlordism extracting 50-55% in-kind rice rent under soaring military procurement prices.',
            'mechanism': '1917 rice price index surges to 1.22 due to Siberian military stockpiling and speculative cornering; landlords enforce strict autumn in-kind rent and demand tenancy re-contracting (+2 rin).',
            'village_impact': 'Cartel-Lock leaves impoverished tenant with barely £8.94/t net rice value, depressing marketed share to 0.495; Cartel-Split allows direct farmer sale to urban millers, lifting farmgate to £14.88/t (+66.4%) and raising sold_share to 0.612.',
        },
    )


# =========================================================================
# 1917→1920 WINDOW SUPPORT — p_world anchors beyond the 1914-1917 baseline
# =========================================================================
# The engine's native window is 1914-01..1917-01 (anchor year 1914, 36 months).
# The 1920 observation layer needs the window 1917-01..1920-01 (37 points), which
# reads p_world anchors 1917..1920. The 1920 values below are the committed canon
# from the ECC 1920 projection suite (ECC scripts/build-1920-bifurcation-suite.py
# market_specs p_world_1920, consumed by 1920/projection/dynamics-gnp.json,
# [DERIVED_PROXY], SSOT＝該檔). 1918/1919 are mechanical linear midpoints of the
# 1917→1920 ramp — no new economic judgement is introduced. The default 1914
# window only reads keys 1914..1917, so extending the series is a no-op for it.
P_WORLD_1920_CANON: Dict[str, float] = {
    'ar_frigorifico': 110.0,
    'us_midwest_grain': 108.0,
    'uk_corn_factor': 115.0,
    'in_chettiar_credit': 105.0,
    'us_furnish_lien': 120.0,
    'prd_silk_filature': 72.0,
    'jp_rice_tenancy': 118.0,
}


def extend_p_world_to_1920(cfg: MarketConfig) -> MarketConfig:
    """Attach [DERIVED_PROXY] p_world anchors 1918/1919/1920 to a market config.

    1918 and 1919 are exact thirds of the 1917→1920 canon ramp, so the monthly
    p_world path over the 1917 window is a single linear glide from the committed
    1917 anchor to the committed 1920 canon value. No-op for the default 1914
    window (those months only ever read keys 1914..1917).
    """
    p17 = cfg.p_world_series[1917]
    p20 = P_WORLD_1920_CANON[cfg.market_id]
    cfg.p_world_series = dict(cfg.p_world_series)
    cfg.p_world_series[1918] = round(p17 + (p20 - p17) / 3.0, 4)
    cfg.p_world_series[1919] = round(p17 + 2.0 * (p20 - p17) / 3.0, 4)
    cfg.p_world_series[1920] = p20
    return cfg


def get_all_market_configs() -> Dict[str, MarketConfig]:
    return {
        'ar_frigorifico': extend_p_world_to_1920(get_ar_frigorifico_config()),
        'us_midwest_grain': extend_p_world_to_1920(get_us_midwest_grain_config()),
        'uk_corn_factor': extend_p_world_to_1920(get_uk_corn_factor_config()),
        'in_chettiar_credit': extend_p_world_to_1920(get_in_chettiar_credit_config()),
        'us_furnish_lien': extend_p_world_to_1920(get_us_furnish_lien_config()),
        'prd_silk_filature': extend_p_world_to_1920(get_prd_silk_filature_config()),
        'jp_rice_tenancy': extend_p_world_to_1920(get_jp_rice_tenancy_config()),
    }


def run_and_save_all_markets(
    output_dir: Path,
    root_path: Optional[Path] = None,
    delta_idle: float = 0.20,
    save_village_monthly: bool = False,
    start_year: int = 1914,
) -> Dict[str, Any]:
    """Run simulation for all 7 markets with 120-village feedback loop and save outputs.

    ``start_year`` selects the engine window: 1914 (default, legacy
    1914-01..1917-01) or 1917 (1920 observation layer, 1917-01..1920-01). Terminal-
    year keys in the outputs follow the window (sold_share_1917_* vs
    sold_share_1920_*, macro_scorecard_1917 vs macro_scorecard_1920, ...).

    When ``save_village_monthly`` is True, additionally writes one
    ``village-monthly-{market}.json`` per market (per-village monthly sold_share
    series, schema draft s-trajectory-gnp-village-monthly-v1). Default False so
    existing callers keep producing exactly the artifacts they produced before.
    """
    terminal_year = start_year + 3
    root_path = _require_root(root_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all 120 villages
    all_market_villages = load_all_gnp_villages(root_path)
    total_villages = sum(len(v) for v in all_market_villages.values())
    assert total_villages == 120, f'Expected 120 villages, found {total_villages}'

    configs = get_all_market_configs()
    market_results: Dict[str, Any] = {}

    # Track macro distribution shift across all 120 villages
    all_village_shifts = []

    for mid, cfg in configs.items():
        v_list = all_market_villages.get(mid, [])
        cfg.delta_idle = delta_idle
        engine = MarketDynamicsEngine(cfg, villages=v_list)
        suite = engine.run_comparative_suite(delta_idle_override=delta_idle, start_year=start_year)

        # Save individual market file
        out_file = output_dir / f'dynamics-gnp-{mid.replace("_", "-")}.json'
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(suite, f, indent=2, ensure_ascii=False)
        _log = out_file.relative_to(root_path) if out_file.is_relative_to(root_path) else out_file
        print(f'[OK] Generated {mid} ({len(v_list)} villages) -> {_log}')
        market_results[mid] = suite

        if save_village_monthly:
            vm_doc = engine.build_village_monthly_document(
                suite=suite, start_year=start_year
            )
            if not vm_doc['endpoint_consistency']['passed']:
                raise AssertionError(
                    f"village-monthly endpoint consistency failed for {mid}: "
                    f"max_abs_deviation={vm_doc['endpoint_consistency']['max_abs_deviation']}"
                )
            vm_file = output_dir / f'village-monthly-{mid.replace("_", "-")}.json'
            with open(vm_file, 'w', encoding='utf-8') as f:
                json.dump(vm_doc, f, indent=2, ensure_ascii=False)
            _log = vm_file.relative_to(root_path) if vm_file.is_relative_to(root_path) else vm_file
            print(f'[OK] Generated village-monthly {mid} ({len(v_list)} villages) -> {_log}')

        for vr in suite['village_transmission']['villages']:
            all_village_shifts.append({
                'village_id': vr['village_id'],
                'market_id': mid,
                'sold_share_1914': vr['sold_share_1914'],
                f'sold_share_{terminal_year}_free': vr[f'sold_share_{terminal_year}_free'],
                f'sold_share_{terminal_year}_lock': vr[f'sold_share_{terminal_year}_lock'],
                f'sold_share_{terminal_year}_split': vr[f'sold_share_{terminal_year}_split'],
            })

    # Calculate 120-village macro metrics
    macro_14 = round(sum(x['sold_share_1914'] for x in all_village_shifts) / len(all_village_shifts), 4)
    macro_t_free = round(sum(x[f'sold_share_{terminal_year}_free'] for x in all_village_shifts) / len(all_village_shifts), 4)
    macro_t_lock = round(sum(x[f'sold_share_{terminal_year}_lock'] for x in all_village_shifts) / len(all_village_shifts), 4)
    macro_t_split = round(sum(x[f'sold_share_{terminal_year}_split'] for x in all_village_shifts) / len(all_village_shifts), 4)

    # Build consolidated document
    consolidated_doc = {
        'schema': 'dynamics-gnp-multi-market-consolidated',
        'canon_ruling': 'SINGLE_CANON_DIRECT_ADJUSTMENT',
        'framework': 'SLOT-MIRROR-SCHEMA §六 + Lenin 1916 Monopoly-Competition Coexistence + Closed-Loop Feedback',
        'total_markets': len(market_results),
        'total_gnp_villages_audited': total_villages,
        'delta_idle_baseline': delta_idle,
        'parameter_tagging': '[DERIVED_PROXY]',
        f'macro_scorecard_{terminal_year}': {
            'mean_sold_share_1914_baseline': macro_14,
            f'mean_sold_share_{terminal_year}_free_evolution': macro_t_free,
            f'mean_sold_share_{terminal_year}_cartel_lock': macro_t_lock,
            f'mean_sold_share_{terminal_year}_cartel_split': macro_t_split,
            'macro_bifurcation_split_vs_lock': round(macro_t_split - macro_t_lock, 4),
            'macro_retention_suppression_lock_vs_free': round(macro_t_lock - macro_t_free, 4),
        },
        'markets': {},
    }

    for mid, res in market_results.items():
        free_ann = res['regimes']['free_evolution']['annual_readouts']
        lock_ann = res['regimes']['cartel_lock']['annual_readouts']
        split_ann = res['regimes']['cartel_split']['annual_readouts']

        consolidated_doc['markets'][mid] = {
            'commodity': res['commodity'],
            'geography': res['geography'],
            'provenance': res['provenance'],
            'total_villages': res['village_transmission']['total_villages'],
            'kappa_trajectory': {
                yr: ro['kappa'] for yr, ro in free_ann.items()
            },
            'sold_share_trajectory': {
                yr: ro['village_sold_share'] for yr, ro in free_ann.items()
            },
            f'bifurcation_{terminal_year}': {
                'kappa': {
                    'free_evolution': free_ann[str(terminal_year)]['kappa'],
                    'cartel_split': split_ann[str(terminal_year)]['kappa'],
                    'cartel_lock': lock_ann[str(terminal_year)]['kappa'],
                    'delta_split_vs_lock': round(split_ann[str(terminal_year)]['kappa'] - lock_ann[str(terminal_year)]['kappa'], 4),
                },
                'sold_share': {
                    'free_evolution': free_ann[str(terminal_year)]['village_sold_share'],
                    'cartel_split': split_ann[str(terminal_year)]['village_sold_share'],
                    'cartel_lock': lock_ann[str(terminal_year)]['village_sold_share'],
                    'bifurcation_delta_sold_share': round(
                        split_ann[str(terminal_year)]['village_sold_share'] - lock_ann[str(terminal_year)]['village_sold_share'], 4
                    ),
                },
                'farmgate_price': {
                    'free_evolution': free_ann[str(terminal_year)]['p_farmgate'],
                    'cartel_split': split_ann[str(terminal_year)]['p_farmgate'],
                    'cartel_lock': lock_ann[str(terminal_year)]['p_farmgate'],
                    'farmgate_premium_split_vs_lock': round(
                        split_ann[str(terminal_year)]['p_farmgate'] - lock_ann[str(terminal_year)]['p_farmgate'], 2
                    ),
                },
            },
            'village_transmission_summary': res['village_transmission']['village_panel_summary'],
            'lenin_thesis': res['bifurcation_summary']['lenin_proposition_readout'],
        }

    cons_path = output_dir / 'dynamics-gnp-markets-consolidated.json'
    with open(cons_path, 'w', encoding='utf-8') as f:
        json.dump(consolidated_doc, f, indent=2, ensure_ascii=False)
    _log = cons_path.relative_to(root_path) if cons_path.is_relative_to(root_path) else cons_path
    print(f'[OK] Generated Consolidated Suite -> {_log}')

    return {
        'market_results': market_results,
        'consolidated_doc': consolidated_doc,
        'all_village_shifts': all_village_shifts,
    }


def run_and_save_village_monthly(
    output_dir: Path,
    root_path: Optional[Path] = None,
    delta_idle: float = 0.20,
    start_year: int = 1914,
) -> Dict[str, Dict[str, Any]]:
    """Serialize per-village monthly sold_share series for all 7 GNP markets.

    Writes one ``village-monthly-{market}.json`` per market (schema draft
    ``s-trajectory-gnp-village-monthly-v1``) without touching the existing
    ``dynamics-gnp-{market}.json`` artifacts. Endpoint consistency against the
    annual three-regime village values is asserted per market (tolerance 1e-6).
    ``start_year`` selects the window (1914 legacy / 1917 for the 1920 layer).
    """
    root_path = _require_root(root_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_market_villages = load_all_gnp_villages(root_path)
    total_villages = sum(len(v) for v in all_market_villages.values())
    assert total_villages == 120, f'Expected 120 villages, found {total_villages}'

    configs = get_all_market_configs()
    docs: Dict[str, Dict[str, Any]] = {}

    for mid, cfg in configs.items():
        v_list = all_market_villages.get(mid, [])
        cfg.delta_idle = delta_idle
        engine = MarketDynamicsEngine(cfg, villages=v_list)
        suite = engine.run_comparative_suite(delta_idle_override=delta_idle, start_year=start_year)
        doc = engine.build_village_monthly_document(suite=suite, start_year=start_year)
        assert doc['endpoint_consistency']['passed'], (
            f"village-monthly endpoint consistency failed for {mid}: "
            f"max_abs_deviation={doc['endpoint_consistency']['max_abs_deviation']}"
        )
        out_file = output_dir / f'village-monthly-{mid.replace("_", "-")}.json'
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        _log = out_file.relative_to(root_path) if out_file.is_relative_to(root_path) else out_file
        print(f'[OK] Generated village-monthly {mid} ({len(v_list)} villages) -> {_log}')
        docs[mid] = doc

    return docs


def run_sensitivity_scan(delta_idles: List[float] = [0.0, 0.20, 0.40], root_path: Optional[Path] = None) -> Dict[str, Any]:
    """Run sensitivity analysis of the delta_idle feedback penalty parameter."""
    root_path = _require_root(root_path)

    all_market_villages = load_all_gnp_villages(root_path)
    configs = get_all_market_configs()
    
    results = {}
    for d in delta_idles:
        market_readouts = {}
        all_village_ss = []
        for mid, cfg in configs.items():
            v_list = all_market_villages.get(mid, [])
            cfg.delta_idle = d
            engine = MarketDynamicsEngine(cfg, villages=v_list)
            res = engine.run_simulation(mode='free_evolution', delta_idle_override=d)
            y17 = res['annual_readouts']['1917']
            market_readouts[mid] = {
                'kappa_1917': y17['kappa'],
                'effective_sold_share_1917': y17['village_sold_share'],
                'p_farmgate_1917': y17['p_farmgate'],
                'penalty_idle_1917': y17['penalty_idle'],
            }
            for v in v_list:
                all_village_ss.append(res['village_shares_final'].get(v.village_id, v.sold_share_1914))
                
        macro_ss = round(sum(all_village_ss) / len(all_village_ss), 4) if all_village_ss else 0.0
        avg_kappa = round(sum(m['kappa_1917'] for m in market_readouts.values()) / len(market_readouts), 4)
        
        results[f'delta_idle_{d:.2f}'] = {
            'delta_idle': d,
            'macro_mean_kappa_1917': avg_kappa,
            'macro_mean_sold_share_1917': macro_ss,
            'markets': market_readouts,
        }
        
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run GNP Multi-Market Oligopoly Dynamics Closed-Loop Suite.')
    parser.add_argument(
        '--delta-idle',
        type=float,
        default=0.20,
        help='Capacity idling penalty parameter (default: 0.20 [DERIVED_PROXY])'
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Worldline data root (contains index/locations/data/almanac)"
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=1914,
        help='Engine window start year: 1914 (default, legacy 1914-01..1917-01) '
             'or 1917 (1920 observation layer, 1917-01..1920-01).'
    )
    parser.add_argument(
        '--save-village-monthly',
        action='store_true',
        help='Also write village-monthly-{market}.json per market (opt-in).'
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default=None,
        help='Output directory (default: the legacy 1917 projection dir).'
    )
    args = parser.parse_args()

    root = Path(args.root)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.start_year == 1917:
        out_dir = root / 'index/locations/data/almanac/1920-1929/1920/projection'
    else:
        out_dir = root / 'index/locations/data/almanac/1910-1920/1917/projection'
    print(f'=== EXECUTING COMPLETE 7-MARKET CLOSED-LOOP SUITE (window {args.start_year}-01..{args.start_year + 3}-01) ===')
    run_and_save_all_markets(
        out_dir,
        root_path=root,
        delta_idle=args.delta_idle,
        save_village_monthly=args.save_village_monthly,
        start_year=args.start_year,
    )


if __name__ == '__main__':
    main()
