"""Spectrum OS Kernel — GNP Multi-Actor Oligopoly Dynamics Engine.

Methodology:
- SLOT-MIRROR-SCHEMA.md §六 (多主體 dynamics 軌道與命名裁定):
  GNP = Great National Powers (β₁ 大國集團多極譜系).
  actors_n = N, with κ(t) collusion index in [0, 1] as the core state variable:
    κ -> 1: Cartel quota / monopsony collusion (organized capitalism, Hilferding general cartel)
    κ -> 0: Cartel split / cutthroat competition / price war
  Counterfactual track: cartel-split.
- THEORY-LEDGER.md (lines 1005-1050):
  7 empirical actor cards extracted from the 360 observation-gnp volumes.
  Parameters grounded in empirical proxy measurements with [DERIVED_PROXY] tagging.
- Integrates with spectrum_os.kernel.forces.ForceFieldDynamics for resultant S_zone(t) = R/(R+C).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from spectrum_os.kernel.forces import ForceFieldDynamics, ForceTrajectory


@dataclass
class ActorCard:
    """Empirical capitalist oligarch actor card in the GNP worldline."""
    actor_id: str
    name: str
    bloc: str
    sphere: str
    actor_type: str
    source_volumes: List[str]
    parameters: Dict[str, Any]
    market_power: float
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "name": self.name,
            "bloc": self.bloc,
            "sphere": self.sphere,
            "actor_type": self.actor_type,
            "source_volumes": self.source_volumes,
            "parameters": self.parameters,
            "market_power": round(float(self.market_power), 4),
            "description": self.description,
        }


@dataclass
class CommodityMarket:
    """A regional commodity export market contested by N oligarch actors."""
    market_id: str
    commodity: str
    terminal_market: str
    actor_ids: List[str]
    actor_shares: Dict[str, float]
    base_terminal_price_gbp: float
    cost_freight_gbp: float
    cost_processing_gbp: Dict[str, float]
    monopsony_markdown: float
    initial_kappa: float
    gamma_cartel: float = 0.12     # Rate of cartel quota discipline convergence
    delta_split: float = 0.85      # Rate of collusion collapse under cartel-split
    theta_shock: float = 0.20      # Sensitivity of collusion to world price shocks
    price_cycle_index: Dict[str, float] = field(default_factory=dict)
    linked_nodes: List[str] = field(default_factory=list)

    @property
    def hhi(self) -> float:
        """Herfindahl-Hirschman Index of market concentration."""
        return sum(s ** 2 for s in self.actor_shares.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_id": self.market_id,
            "commodity": self.commodity,
            "terminal_market": self.terminal_market,
            "actor_ids": self.actor_ids,
            "actor_shares": {k: round(v, 4) for k, v in self.actor_shares.items()},
            "hhi": round(self.hhi, 4),
            "initial_kappa": round(self.initial_kappa, 4),
            "base_terminal_price_gbp": self.base_terminal_price_gbp,
            "cost_freight_gbp": self.cost_freight_gbp,
            "cost_processing_gbp": self.cost_processing_gbp,
            "monopsony_markdown": self.monopsony_markdown,
            "parameters_provenance": "[DERIVED_PROXY] from observation-gnp volumes",
            "linked_nodes": self.linked_nodes,
        }


class GNPOligopolyDynamics:
    """N-actor multi-oligarch dynamics engine for GNP worldline."""

    def __init__(self, dt: float = 1.0 / 12.0, total_months: int = 36):
        self.dt = dt
        self.total_months = total_months
        self.actor_registry: Dict[str, ActorCard] = {}
        self.market_registry: Dict[str, CommodityMarket] = {}

    def register_actor(self, actor: ActorCard) -> None:
        self.actor_registry[actor.actor_id] = actor

    def register_market(self, market: CommodityMarket) -> None:
        self.market_registry[market.market_id] = market

    def compute_price_trajectory(
        self,
        market: CommodityMarket,
        price_cycle_1917_1920: Optional[List[float]] = None
    ) -> np.ndarray:
        """Interpolate or generate monthly terminal export price index."""
        if price_cycle_1917_1920 is not None and len(price_cycle_1917_1920) == self.total_months + 1:
            index_series = np.array(price_cycle_1917_1920, dtype=np.float64)
        else:
            p1917 = market.price_cycle_index.get("1917", 131.0)
            p1920 = market.price_cycle_index.get("1920", 112.0)
            t_steps = np.linspace(0.0, 1.0, self.total_months + 1)
            index_series = p1917 + (p1920 - p1917) * 0.5 * (1.0 - np.cos(np.pi * t_steps))

        p_terminal = market.base_terminal_price_gbp * (index_series / 100.0)
        return p_terminal

    def simulate_market(
        self,
        market_id: str,
        cartel_split: bool = False,
        custom_price_cycle: Optional[List[float]] = None,
        node_initial_states: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """Run 36-month monthly simulation of commodity market & linked rural nodes."""
        if market_id not in self.market_registry:
            raise KeyError(f"Market '{market_id}' not found in registry")

        market = self.market_registry[market_id]
        p_terminal = self.compute_price_trajectory(market, custom_price_cycle)

        n_steps = self.total_months + 1
        kappa = np.zeros(n_steps, dtype=np.float64)
        p_farmgate = np.zeros(n_steps, dtype=np.float64)
        retention = np.zeros(n_steps, dtype=np.float64)
        actor_margins = {aid: np.zeros(n_steps, dtype=np.float64) for aid in market.actor_ids}

        kappa[0] = market.initial_kappa
        split_month = 2 if cartel_split else 9999

        c_freight = market.cost_freight_gbp
        avg_proc = float(np.mean(list(market.cost_processing_gbp.values())))
        c_normal_profit = 2.50  # [DERIVED_PROXY] normal trading return

        for t in range(n_steps):
            cur_p_term = p_terminal[t]
            p_comp = max(cur_p_term - c_freight - avg_proc - c_normal_profit, 1.0)
            p_mono = p_comp * (1.0 - market.monopsony_markdown)

            # Effective farmgate price
            p_farm = (1.0 - kappa[t]) * p_comp + kappa[t] * p_mono
            p_farmgate[t] = p_farm
            retention[t] = p_farm / cur_p_term

            # Oligarch margins
            for aid in market.actor_ids:
                c_proc_k = market.cost_processing_gbp.get(aid, avg_proc)
                margin_k = cur_p_term - p_farm - c_freight - c_proc_k
                actor_margins[aid][t] = margin_k

            # Evolve kappa for next step
            if t < self.total_months:
                price_diff_rel = abs(p_terminal[t + 1] - p_terminal[t]) / max(p_terminal[t], 1e-3)
                if cartel_split and t >= split_month:
                    d_kappa = -market.delta_split * kappa[t]
                else:
                    d_kappa = market.gamma_cartel * kappa[t] * (1.0 - kappa[t]) - market.theta_shock * price_diff_rel

                next_k = kappa[t] + d_kappa * self.dt
                kappa[t + 1] = float(np.clip(next_k, 0.05, 0.95))

        # Rural Node Force Field Integration
        node_trajectories = {}
        if node_initial_states is None:
            node_initial_states = {
                nid: {"R_0": 0.85, "C_0": 0.55, "P_R0": 0.10}
                for nid in market.linked_nodes
            }

        p_farm_baseline_0 = p_farmgate[0]
        for nid, init_state in node_initial_states.items():
            r0 = init_state.get("R_0", 0.85)
            c0 = init_state.get("C_0", 0.55)
            pr0 = init_state.get("P_R0", 0.10)

            engine = ForceFieldDynamics(
                revolutionary_force=r0,
                conservative_force=c0,
                latent_force=pr0,
                growth_r=0.02,
                growth_c=0.01,
                alpha=0.05,
                beta=0.03,
                mu=0.02,
                lam=0.15,
                dt=self.dt,
                method="rk4",
            )
            r_seq = np.zeros(n_steps)
            c_seq = np.zeros(n_steps)
            s_seq = np.zeros(n_steps)
            t_seq = np.zeros(n_steps)

            r_seq[0] = float(engine._R[0])
            c_seq[0] = float(engine._C[0])
            s_seq[0] = float(engine.s()[0])
            t_seq[0] = float(engine.total_capacity()[0])

            for m in range(self.total_months):
                price_ratio = p_farmgate[m] / p_farm_baseline_0
                eta_price = 0.08  # [DERIVED_PROXY] price pass-through elasticity to rural R
                r_drive = eta_price * (price_ratio - 1.0)

                eta_cartel = 0.06  # [DERIVED_PROXY] cartel rent extraction drag on rural C
                c_drag = eta_cartel * (kappa[m] - market.initial_kappa)

                engine.step()
                # Apply market external field injections
                engine._R[0] = max(float(engine._R[0] + r_drive * self.dt), 0.01)
                engine._C[0] = max(float(engine._C[0] + c_drag * self.dt), 0.01)

                r_seq[m + 1] = float(engine._R[0])
                c_seq[m + 1] = float(engine._C[0])
                s_seq[m + 1] = float(engine.s()[0])
                t_seq[m + 1] = float(engine.total_capacity()[0])

            node_trajectories[nid] = {
                "S_1917": round(float(s_seq[0]), 4),
                "S_1920": round(float(s_seq[-1]), 4),
                "R_1917": round(float(r_seq[0]), 4),
                "R_1920": round(float(r_seq[-1]), 4),
                "C_1917": round(float(c_seq[0]), 4),
                "C_1920": round(float(c_seq[-1]), 4),
                "T_1920": round(float(t_seq[-1]), 4),
                "delta_S": round(float(s_seq[-1] - s_seq[0]), 4),
                "delta_R": round(float(r_seq[-1] - r_seq[0]), 4),
                "delta_C": round(float(c_seq[-1] - c_seq[0]), 4),
                "farmgate_price_1917": round(float(p_farmgate[0]), 2),
                "farmgate_price_1920": round(float(p_farmgate[-1]), 2),
                "retention_rate_1917": round(float(retention[0]), 4),
                "retention_rate_1920": round(float(retention[-1]), 4),
            }

        return {
            "mode": "cartel_split_counterfactual" if cartel_split else "baseline_collusion",
            "market_id": market_id,
            "total_months": self.total_months,
            "monthly_series": {
                "months": [f"M_{i:02d}" for i in range(n_steps)],
                "kappa": [round(float(k), 4) for k in kappa],
                "p_terminal_gbp": [round(float(p), 2) for p in p_terminal],
                "p_farmgate_gbp": [round(float(p), 2) for p in p_farmgate],
                "retention_rate": [round(float(r), 4) for r in retention],
                "actor_margins_gbp": {
                    aid: [round(float(m), 2) for m in m_arr]
                    for aid, m_arr in actor_margins.items()
                },
            },
            "summary_1917_vs_1920": {
                "kappa_1917": round(float(kappa[0]), 4),
                "kappa_1920": round(float(kappa[-1]), 4),
                "p_terminal_1917": round(float(p_terminal[0]), 2),
                "p_terminal_1920": round(float(p_terminal[-1]), 2),
                "p_farmgate_1917": round(float(p_farmgate[0]), 2),
                "p_farmgate_1920": round(float(p_farmgate[-1]), 2),
                "retention_1917": round(float(retention[0]), 4),
                "retention_1920": round(float(retention[-1]), 4),
                "margins_1917": {aid: round(float(m[0]), 2) for aid, m in actor_margins.items()},
                "margins_1920": {aid: round(float(m[-1]), 2) for aid, m in actor_margins.items()},
            },
            "node_trajectories": node_trajectories,
        }
