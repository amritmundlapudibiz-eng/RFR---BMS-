"""
balancer.py — Active Cell Balancing (Passive Bleeding Strategy)
Rutgers Formula Racing BMS Fault Detector

When the voltage spread across the 96-series pack exceeds a configurable
threshold (default 50 mV), identifies cells above the target voltage and
applies a simulated bleed current through a discharge resistor.

Each balancing event (start → end) is recorded with cell ID, voltage delta,
bleed current, and duration for CSV logging.

In a real accumulator this would drive a MOSFET + bleed resistor per cell
on the LTC6813 / BQ76952 slave board.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from simulator import PackState, CellState, SERIES_COUNT, PARALLEL_COUNT, CELL_CAPACITY_AH


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BalancerConfig:
    threshold_v:      float = 0.050   # activate when pack ΔV > 50 mV
    target_strategy:  str   = "median" # "median" | "min" — bleed toward this
    bleed_resistance: float = 33.0    # Ω  (physical bleed resistor)
    min_bleed_delta:  float = 0.010   # V — don't bleed if Δ < 10 mV
    min_cell_voltage: float = 3.30    # V — never bleed below this
    hysteresis_v:     float = 0.010   # V — stop bleeding 10 mV before target


# ---------------------------------------------------------------------------
# Balancing event record
# ---------------------------------------------------------------------------

@dataclass
class BalancingEvent:
    """One completed or in-progress balancing action on a single series group."""
    series_index:      int
    start_time:        float
    start_voltage:     float         # cell voltage when bleed started
    target_voltage:    float         # voltage we're bleeding toward
    voltage_delta:     float         # start_voltage - target at activation
    bleed_current_ma:  float         # mA through bleed resistor
    duration_s:        float = 0.0
    end_voltage:       float = 0.0
    completed:         bool  = False


# ---------------------------------------------------------------------------
# Cell Balancer
# ---------------------------------------------------------------------------

class CellBalancer:
    """
    Passive-bleeding cell balancer for a series-connected LiPo pack.

    Call evaluate() each simulation step.  Returns a dict of
    {series_index: bleed_current_A} to be applied by the simulator.
    """

    def __init__(self, config: Optional[BalancerConfig] = None):
        self.cfg = config or BalancerConfig()

        # Currently active bleeds:  series_index → BalancingEvent
        self.active_bleeds: Dict[int, BalancingEvent] = {}

        # Completed events (for logging / review)
        self.completed_events: List[BalancingEvent] = []

        # Counters
        self.total_events:     int = 0
        self.total_energy_mwh: float = 0.0   # total energy dissipated

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, state: PackState, dt: float) -> Dict[int, float]:
        """
        Evaluate pack balance and return bleed commands.

        Returns
        -------
        bleed_plan : dict
            {series_index: bleed_current_A}  — current to sink from each cell
            group.  Empty dict = no balancing needed.
        """
        # Compute per-series-group average voltage
        series_v = self._series_voltages(state)
        v_values = list(series_v.values())
        v_spread = max(v_values) - min(v_values)

        bleed_plan: Dict[int, float] = {}

        if v_spread > self.cfg.threshold_v:
            # Determine target voltage
            if self.cfg.target_strategy == "median":
                v_target = float(np.median(v_values))
            else:
                v_target = min(v_values)

            for s_idx, v in series_v.items():
                delta = v - v_target

                if delta > self.cfg.min_bleed_delta and v > self.cfg.min_cell_voltage:
                    # Calculate bleed current  I = V / R
                    i_bleed = v / self.cfg.bleed_resistance   # A

                    bleed_plan[s_idx] = i_bleed

                    # Start tracking new bleed event
                    if s_idx not in self.active_bleeds:
                        self.active_bleeds[s_idx] = BalancingEvent(
                            series_index    = s_idx,
                            start_time      = state.timestamp,
                            start_voltage   = v,
                            target_voltage  = v_target,
                            voltage_delta   = delta,
                            bleed_current_ma = i_bleed * 1000.0,
                        )
                    else:
                        # Update duration on existing bleed
                        self.active_bleeds[s_idx].duration_s += dt

                    # Accumulate energy  E = I * V * dt  (Wh)
                    self.total_energy_mwh += i_bleed * v * dt / 3.6  # mWh

        # Close out bleeds that are no longer active
        for s_idx in list(self.active_bleeds.keys()):
            if s_idx not in bleed_plan:
                event = self.active_bleeds.pop(s_idx)
                event.completed   = True
                event.end_voltage = series_v.get(s_idx, event.start_voltage)
                event.duration_s  = max(event.duration_s, 0.001)
                self.completed_events.append(event)
                self.total_events += 1

        return bleed_plan

    def apply_bleed(
        self,
        state: PackState,
        bleed_plan: Dict[int, float],
        dt: float,
    ) -> None:
        """
        Apply bleed currents to cells in the pack state.

        For each series group with a bleed command, reduce the SOC and
        voltage of all parallel cells by the bleed current's effect.
        """
        for s_idx, i_bleed in bleed_plan.items():
            for cell in state.cells:
                if cell.series_index == s_idx and cell.injected_fault is None:
                    # SOC reduction from bleed: ΔQ = I_bleed * dt
                    dsoc = i_bleed * dt / (CELL_CAPACITY_AH * 3600.0)
                    cell.soc = max(0.0, cell.soc - dsoc)

                    # Voltage drop from bleed (small but visible)
                    cell.voltage -= i_bleed * cell.internal_resistance

                    # Tiny thermal contribution from bleed resistor
                    cell.temperature += (i_bleed ** 2 * self.cfg.bleed_resistance
                                         * dt / 50.0 * 0.001)

    @property
    def is_balancing(self) -> bool:
        return len(self.active_bleeds) > 0

    @property
    def active_count(self) -> int:
        return len(self.active_bleeds)

    def get_new_completed(self) -> List[BalancingEvent]:
        """Pop and return completed events (for logging)."""
        events = self.completed_events.copy()
        self.completed_events.clear()
        return events

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _series_voltages(state: PackState) -> Dict[int, float]:
        """Average voltage per series group."""
        groups: Dict[int, List[float]] = defaultdict(list)
        for c in state.cells:
            groups[c.series_index].append(c.voltage)
        return {s: float(np.mean(vs)) for s, vs in groups.items()}
