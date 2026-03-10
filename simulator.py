"""
simulator.py — 96s2p LiPo Pack Physics Simulator
Rutgers Formula Racing BMS Fault Detector

Models cell voltages, temperatures, and current with realistic electrochemistry
for a 96-series, 2-parallel LiPo accumulator (~355 V nominal, 10 Ah).
"""

import random
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Pack geometry
# ---------------------------------------------------------------------------
SERIES_COUNT = 96   # Series cell groups
PARALLEL_COUNT = 2    # Parallel strings per series group
TOTAL_CELLS = SERIES_COUNT * PARALLEL_COUNT  # 192 cells

# ---------------------------------------------------------------------------
# Cell parameters (typical 5 Ah LiPo pouch)
# ---------------------------------------------------------------------------
CELL_CAPACITY_AH = 5.0    # Ah per cell (10 Ah per parallel group)
CELL_INTERNAL_RESISTANCE = 0.015  # Ω — nominal; ±5 % spread added at init
CELL_THERMAL_MASS = 50.0   # J/K
CELL_THERMAL_CONDUCTANCE = 0.5    # W/K  (cell → coolant/ambient)

# LiPo OCV vs SOC lookup (SOC 0 → 1)
_SOC_PTS = np.array([0.00, 0.05, 0.10, 0.20, 0.30, 0.40,
                     0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00])
_OCV_PTS = np.array([2.50, 2.80, 3.00, 3.40, 3.60, 3.70,
                     3.75, 3.80, 3.85, 3.90, 4.00, 4.10, 4.20])

# Load profile → pack current (A)  [positive = discharge]
LOAD_CURRENT_MAP = {
    "idle":         2.0,
    "cruising":    50.0,
    "accelerating": 150.0,
    "braking": -40.0,   # regenerative braking
    "charging": -15.0,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CellState:
    """State of a single cell in the pack."""
    cell_id:       int
    series_index:  int          # 0 – 95
    parallel_index: int         # 0 – 1

    soc:                float = 0.80
    voltage:            float = 3.90   # terminal voltage (V)
    temperature:        float = 25.0   # °C
    current:            float = 0.0    # A, positive = discharge

    internal_resistance: float = CELL_INTERNAL_RESISTANCE
    # State of Health (%), updated by SohEstimator
    soh:                 float = 100.0
    injected_fault:      Optional[str] = None   # None or fault label


@dataclass
class PackState:
    """Snapshot of the full 96s2p pack."""
    cells: List[CellState]

    pack_current: float = 0.0   # A (positive = discharge)
    pack_voltage: float = 0.0   # V

    # IMD model: isolation resistance between HV rails and chassis (MΩ)
    iso_resistance_pos: float = 10.0   # + rail → chassis
    iso_resistance_neg: float = 10.0   # − rail → chassis

    airs_closed:    bool = True   # Accumulator Isolation Relays
    imd_triggered:  bool = False  # IMD fault latch
    timestamp:      float = field(default_factory=time.time)

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------
    @property
    def min_cell_voltage(self) -> float:
        return float(min(c.voltage for c in self.cells))

    @property
    def max_cell_voltage(self) -> float:
        return float(max(c.voltage for c in self.cells))

    @property
    def min_cell_temp(self) -> float:
        return float(min(c.temperature for c in self.cells))

    @property
    def max_cell_temp(self) -> float:
        return float(max(c.temperature for c in self.cells))

    @property
    def avg_cell_voltage(self) -> float:
        return float(np.mean([c.voltage for c in self.cells]))

    @property
    def avg_cell_temp(self) -> float:
        return float(np.mean([c.temperature for c in self.cells]))

    @property
    def pack_soc(self) -> float:
        return float(np.mean([c.soc for c in self.cells]))

    @property
    def delta_voltage(self) -> float:
        """Max cell voltage spread — early imbalance indicator."""
        voltages = [c.voltage for c in self.cells]
        return float(max(voltages) - min(voltages))


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class PackSimulator:
    """
    Physics-based simulator for a 96s2p LiPo accumulator.

    Each step() call advances the model by `dt` seconds and returns the
    current PackState.  Fault injection methods allow deterministic or
    random fault scenarios for BMS testing.
    """

    def __init__(
        self,
        initial_soc:  float = 0.80,
        ambient_temp: float = 25.0,
        seed:         int = 42,
    ):
        random.seed(seed)
        np.random.seed(seed)

        self.ambient_temp = ambient_temp
        self._lock = threading.Lock()
        self._sim_time = 0.0
        self._last_wall = time.time()

        # Current control
        self._load_mode = "idle"
        self._target_current = LOAD_CURRENT_MAP["idle"]
        self._current_ramp = 80.0   # A/s

        # Pending injected faults: list of dicts
        self._fault_queue: List[dict] = []

        # Build cells
        cells: List[CellState] = []
        for s in range(SERIES_COUNT):
            for p in range(PARALLEL_COUNT):
                soc = float(
                    np.clip(initial_soc + random.gauss(0, 0.005), 0.05, 0.99))
                r = CELL_INTERNAL_RESISTANCE * (1 + random.gauss(0, 0.05))
                c = CellState(
                    cell_id=s * PARALLEL_COUNT + p,
                    series_index=s,
                    parallel_index=p,
                    soc=soc,
                    internal_resistance=max(r, 0.008),
                    temperature=ambient_temp + random.gauss(0, 1.5),
                )
                c.voltage = self._ocv(c.soc)
                cells.append(c)

        self.state = PackState(cells=cells)
        self._refresh_pack_voltage()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_load(self, mode: str) -> None:
        """Switch drive mode: idle | cruising | accelerating | braking | charging"""
        if mode not in LOAD_CURRENT_MAP:
            raise ValueError(
                f"Unknown mode '{mode}'. Valid: {list(LOAD_CURRENT_MAP)}")
        self._load_mode = mode
        self._target_current = LOAD_CURRENT_MAP[mode]

    def inject_fault(
        self,
        fault_type:  str,
        cell_index:  Optional[int] = None,
        delay_s:     float = 0.0,
    ) -> None:
        """
        Schedule a fault injection.

        fault_type: 'overvoltage' | 'undervoltage' | 'overtemp'
                    | 'overcurrent' | 'isolation'
        cell_index: specific cell (0–191); None → random cell selected at trigger
        delay_s:    seconds from now before fault is applied
        """
        self._fault_queue.append({
            "type":         fault_type,
            "cell_index":   cell_index,
            "trigger_time": self._sim_time + delay_s,
        })

    def clear_faults(self) -> None:
        """Remove all injected faults and restore nominal physics."""
        with self._lock:
            for c in self.state.cells:
                if c.injected_fault:
                    c.injected_fault = None
                    c.voltage = self._ocv(c.soc)
                    c.temperature = self.ambient_temp + random.gauss(0, 2.0)
            self.state.iso_resistance_pos = 10.0
            self.state.iso_resistance_neg = 10.0
            self.state.pack_current = self._target_current
            self._fault_queue.clear()

    def step(self, dt: Optional[float] = None) -> PackState:
        """
        Advance the simulation.  Returns the updated PackState.

        dt: time step in seconds; if None, uses real wall-clock delta (capped 500 ms).
        """
        with self._lock:
            now = time.time()
            if dt is None:
                dt = min(now - self._last_wall, 0.5)
            self._last_wall = now
            self._sim_time += dt

            self._ramp_current(dt)
            self._update_cells(dt)
            self._apply_fault_queue()
            self._refresh_pack_voltage()
            self.state.timestamp = now
            return self.state

    def get_state(self) -> PackState:
        """Thread-safe state snapshot (no physics advance)."""
        with self._lock:
            return self.state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ocv(soc: float) -> float:
        return float(np.interp(soc, _SOC_PTS, _OCV_PTS))

    def _ramp_current(self, dt: float) -> None:
        delta = self._target_current - self.state.pack_current
        max_step = self._current_ramp * dt
        self.state.pack_current += float(np.clip(delta, -max_step, max_step))
        self.state.pack_current += random.gauss(0, 0.3)

    def _update_cells(self, dt: float) -> None:
        i_cell = self.state.pack_current / PARALLEL_COUNT  # current per cell

        for c in self.state.cells:
            if c.injected_fault in ("overvoltage", "undervoltage", "overtemp"):
                continue  # held by injector

            c.current = i_cell

            # SOC integration: ΔQ = I·dt (Coulomb counting)
            c.soc = float(np.clip(
                c.soc - c.current * dt / (CELL_CAPACITY_AH * 3600.0),
                0.0, 1.0,
            ))

            # Terminal voltage  V = OCV(SOC) - I·R  + noise
            c.voltage = (
                self._ocv(c.soc)
                - c.current * c.internal_resistance
                + random.gauss(0, 0.002)
            )

            # Thermal model: dT/dt = (P_joule - P_cool) / thermal_mass
            p_joule = c.current ** 2 * c.internal_resistance
            p_cool = CELL_THERMAL_CONDUCTANCE * \
                (c.temperature - self.ambient_temp)
            c.temperature += (p_joule - p_cool) / CELL_THERMAL_MASS * dt
            c.temperature += random.gauss(0, 0.05)  # ADC noise

    def _apply_fault_queue(self) -> None:
        remaining = []
        for fault in self._fault_queue:
            if self._sim_time >= fault["trigger_time"]:
                self._trigger_fault(fault)
            else:
                remaining.append(fault)
        self._fault_queue = remaining

    def _trigger_fault(self, fault: dict) -> None:
        ft = fault["type"]
        ci = fault["cell_index"]

        if ft == "overvoltage":
            target = self._pick_cell(ci)
            target.voltage = round(4.25 + random.uniform(0.0, 0.15), 3)
            target.injected_fault = "overvoltage"

        elif ft == "undervoltage":
            target = self._pick_cell(ci)
            target.voltage = round(2.20 + random.uniform(-0.10, 0.05), 3)
            target.soc = 0.01
            target.injected_fault = "undervoltage"

        elif ft == "overtemp":
            target = self._pick_cell(ci)
            target.temperature = round(63.0 + random.uniform(0.0, 8.0), 1)
            target.injected_fault = "overtemp"

        elif ft == "overcurrent":
            # Drive pack current to 220 A (motor controller runaway)
            self.state.pack_current = 220.0 + random.uniform(0.0, 30.0)
            self._target_current = self.state.pack_current

        elif ft == "isolation":
            # Degrade + rail insulation (e.g., coolant leak)
            self.state.iso_resistance_pos = round(
                random.uniform(0.05, 0.25), 4
            )  # MΩ — well below 500 Ω/V threshold

    def _pick_cell(self, index: Optional[int]) -> CellState:
        if index is not None:
            return self.state.cells[index % TOTAL_CELLS]
        return random.choice(self.state.cells)

    def _refresh_pack_voltage(self) -> None:
        """Sum series-group voltages (average parallel cell voltages per group)."""
        total = 0.0
        for s in range(SERIES_COUNT):
            group = [c for c in self.state.cells if c.series_index == s]
            total += np.mean([c.voltage for c in group])
        self.state.pack_voltage = float(total)
