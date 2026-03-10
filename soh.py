"""
soh.py — State of Health (SOH) Estimator
Rutgers Formula Racing BMS Fault Detector

Tracks per-cell SOH using two independent methods:

  1. **Capacity fade** — derives cycle count from cumulative Ah throughput
     (Coulomb counting) and applies a capacity degradation curve.

  2. **Internal resistance rise (dV/dI)** — estimates R_internal from
     voltage/current transients and compares to the cell's BOL value.

The weaker of the two metrics determines the cell's SOH percentage.
Cells below 80 % SOH are flagged as DEGRADED per FSAE best practice.

To keep the simulator visually interesting, cells are initialized with
a random pre-existing wear distribution so some cells start near the
degradation threshold.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from simulator import PackState, CellState, CELL_CAPACITY_AH


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SohConfig:
    # Capacity fade model
    # 0.04 % per cycle → 80 % at 500 cycles
    capacity_fade_per_cycle: float = 0.0004
    # Resistance rise model
    # 0.1 % per cycle → +50 % at 500 cycles
    r_rise_per_cycle:        float = 0.0010

    # SOH threshold
    degraded_threshold:      float = 80.0      # % — flag cell as degraded

    # dV/dI estimation
    # A — minimum ΔI for valid R estimate
    dvdi_min_di:             float = 0.5
    dvdi_window:             int = 20        # samples to look back for dV/dI

    # Initial wear distribution (for simulation realism)
    # Each cell randomly gets initial_cycles_mean ± initial_cycles_std cycles
    initial_cycles_mean:     float = 120.0
    initial_cycles_std:      float = 60.0

    # Aging acceleration factor — speeds up degradation during sim
    # 1.0 = realistic, 500.0 = noticeable in 60 s
    aging_acceleration:      float = 500.0


# ---------------------------------------------------------------------------
# Per-cell tracking data
# ---------------------------------------------------------------------------

@dataclass
class CellSohData:
    """Internal tracking state for a single cell."""
    cell_id:         int
    initial_r:       float            # BOL internal resistance (Ω)
    cumulative_ah:   float = 0.0      # total Ah throughput (absolute)
    cycle_count:     float = 0.0      # derived from cumulative_ah
    soh_capacity:    float = 100.0    # % from capacity fade model
    soh_resistance:  float = 100.0    # % from R rise model
    soh:             float = 100.0    # combined SOH (weakest link)
    degraded:        bool = False
    estimated_r:     float = 0.0      # latest dV/dI resistance estimate

    # Rolling buffers for dV/dI
    v_history: deque = field(default_factory=lambda: deque(maxlen=100))
    i_history: deque = field(default_factory=lambda: deque(maxlen=100))
    r_estimates: deque = field(default_factory=lambda: deque(maxlen=50))


# ---------------------------------------------------------------------------
# SOH Estimator
# ---------------------------------------------------------------------------

class SohEstimator:
    """
    Per-cell State of Health estimator.

    Call initialize() once with the initial cell list, then update() each
    simulation step.  The estimator modifies cell.soh in place and applies
    simulated R-rise to cell.internal_resistance.
    """

    def __init__(self, config: Optional[SohConfig] = None):
        self.cfg = config or SohConfig()
        self._data: Dict[int, CellSohData] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self, cells: List[CellState]) -> None:
        """
        Set up per-cell tracking.  Call once after simulator is constructed.
        Assigns random initial wear to each cell for visual realism.
        """
        for cell in cells:
            # Random pre-existing wear
            init_cycles = max(0.0, random.gauss(
                self.cfg.initial_cycles_mean,
                self.cfg.initial_cycles_std,
            ))

            data = CellSohData(
                cell_id=cell.cell_id,
                initial_r=cell.internal_resistance,
                cumulative_ah=init_cycles * CELL_CAPACITY_AH * 2.0,
                cycle_count=init_cycles,
            )

            # Pre-compute initial SOH from the pre-existing wear
            self._compute_soh(data)

            # Apply initial R degradation to the cell
            r_factor = 1.0 + data.cycle_count * self.cfg.r_rise_per_cycle
            cell.internal_resistance = data.initial_r * r_factor

            # Write SOH to cell
            cell.soh = data.soh

            self._data[cell.cell_id] = data

    def update(self, state: PackState, dt: float) -> None:
        """
        Called each simulation tick.  Updates cumulative Ah, estimates R
        from dV/dI, and recalculates SOH for every cell.
        """
        for cell in state.cells:
            data = self._data.get(cell.cell_id)
            if data is None:
                continue

            # ----------------------------------------------------------
            # 1. Coulomb counting — accumulate |I| * dt
            # ----------------------------------------------------------
            ah_delta = abs(cell.current) * dt / 3600.0
            ah_delta *= self.cfg.aging_acceleration   # speed up for sim
            data.cumulative_ah += ah_delta
            data.cycle_count = data.cumulative_ah / (CELL_CAPACITY_AH * 2.0)

            # ----------------------------------------------------------
            # 2. dV/dI estimation (online R_internal)
            # ----------------------------------------------------------
            data.v_history.append(cell.voltage)
            data.i_history.append(cell.current)

            if len(data.v_history) >= self.cfg.dvdi_window:
                n = self.cfg.dvdi_window
                dv = data.v_history[-1] - data.v_history[-n]
                di = data.i_history[-1] - data.i_history[-n]

                if abs(di) > self.cfg.dvdi_min_di:
                    r_est = abs(dv / di)
                    r_est = float(np.clip(r_est, 0.005, 0.100))
                    data.r_estimates.append(r_est)
                    data.estimated_r = r_est

            # ----------------------------------------------------------
            # 3. Apply simulated resistance rise to the cell
            # ----------------------------------------------------------
            r_factor = 1.0 + data.cycle_count * self.cfg.r_rise_per_cycle
            cell.internal_resistance = data.initial_r * r_factor

            # ----------------------------------------------------------
            # 4. Compute SOH
            # ----------------------------------------------------------
            self._compute_soh(data)
            cell.soh = data.soh

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @property
    def degraded_cells(self) -> List[CellSohData]:
        """Return list of cells below the degraded threshold."""
        return [d for d in self._data.values() if d.degraded]

    @property
    def min_soh(self) -> float:
        if not self._data:
            return 100.0
        return min(d.soh for d in self._data.values())

    @property
    def avg_soh(self) -> float:
        if not self._data:
            return 100.0
        return float(np.mean([d.soh for d in self._data.values()]))

    @property
    def max_cycles(self) -> float:
        if not self._data:
            return 0.0
        return max(d.cycle_count for d in self._data.values())

    def get_cell_soh(self, cell_id: int) -> Optional[CellSohData]:
        return self._data.get(cell_id)

    def get_all_soh(self) -> List[Tuple[int, float, bool]]:
        """Return [(cell_id, soh_pct, degraded), ...] sorted by cell_id."""
        return sorted(
            [(d.cell_id, d.soh, d.degraded) for d in self._data.values()],
            key=lambda x: x[0],
        )

    def get_series_soh(self) -> Dict[int, float]:
        """Return {series_index: min SOH of parallel cells in group}."""
        from simulator import SERIES_COUNT, PARALLEL_COUNT
        result = {}
        for s in range(SERIES_COUNT):
            group_ids = [s * PARALLEL_COUNT + p for p in range(PARALLEL_COUNT)]
            sohs = [self._data[cid].soh for cid in group_ids if cid in self._data]
            if sohs:
                result[s] = min(sohs)
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_soh(self, data: CellSohData) -> None:
        """Recalculate SOH from capacity fade and resistance rise."""
        # Capacity fade:  SOH = 100 - (cycles × fade_rate × 100)
        data.soh_capacity = max(0.0,
                                100.0 * (1.0 - data.cycle_count *
                                         self.cfg.capacity_fade_per_cycle)
                                )

        # Resistance rise:  SOH = 100 / (1 + cycles × rise_rate)
        r_factor = 1.0 + data.cycle_count * self.cfg.r_rise_per_cycle
        data.soh_resistance = min(100.0, 100.0 / r_factor)

        # Combined: weakest link
        data.soh = round(min(data.soh_capacity, data.soh_resistance), 1)
        data.degraded = data.soh < self.cfg.degraded_threshold
