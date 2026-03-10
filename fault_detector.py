"""
fault_detector.py — BMS & IMD Fault Detection Engine
Rutgers Formula Racing BMS Fault Detector

Implements FSAE EV rules-compliant thresholds:
  - Cell overvoltage / undervoltage
  - Overtemperature
  - Overcurrent (continuous + peak)
  - Isolation fault → IMD shutdown → AIR open

IMD model follows FSAE EV.8.4: minimum insulation resistance 500 Ω/V of
max pack voltage.  For a 96s LiPo at 4.2 V/cell → 403.2 V →  ~200 kΩ min.
We use 500 Ω/V rounded to 0.20 MΩ as the trip threshold.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from simulator import PackState, SERIES_COUNT, PARALLEL_COUNT

# ---------------------------------------------------------------------------
# Fault taxonomy
# ---------------------------------------------------------------------------


class FaultType(Enum):
    OVERVOLTAGE = auto()
    UNDERVOLTAGE = auto()
    OVERTEMPERATURE = auto()
    OVERCURRENT = auto()
    VOLTAGE_IMBALANCE = auto()
    ISOLATION_FAULT = auto()
    COMM_FAULT = auto()   # SPI / LTC6813 communication failure
    # Predictive warnings — issued *before* threshold breach
    PRED_OVERVOLTAGE = auto()
    PRED_UNDERVOLTAGE = auto()
    PRED_OVERTEMP = auto()


FAULT_LABELS: Dict[FaultType, str] = {
    FaultType.OVERVOLTAGE:       "OVERVOLTAGE",
    FaultType.UNDERVOLTAGE:      "UNDERVOLTAGE",
    FaultType.OVERTEMPERATURE:   "OVERTEMP",
    FaultType.OVERCURRENT:       "OVERCURRENT",
    FaultType.VOLTAGE_IMBALANCE: "V_IMBALANCE",
    FaultType.ISOLATION_FAULT:   "ISOLATION",
    FaultType.COMM_FAULT:        "COMM_FAULT",
    FaultType.PRED_OVERVOLTAGE:  "PRED_OV",
    FaultType.PRED_UNDERVOLTAGE: "PRED_UV",
    FaultType.PRED_OVERTEMP:     "PRED_OT",
}

# Severity: 0 = predictive, 1 = warning, 2 = critical (opens AIRs)
FAULT_SEVERITY: Dict[FaultType, int] = {
    FaultType.OVERVOLTAGE:       2,
    FaultType.UNDERVOLTAGE:      2,
    FaultType.OVERTEMPERATURE:   2,
    FaultType.OVERCURRENT:       2,
    FaultType.VOLTAGE_IMBALANCE: 1,
    FaultType.ISOLATION_FAULT:   2,
    # comm failure = critical (can't trust cell data)
    FaultType.COMM_FAULT:        2,
    FaultType.PRED_OVERVOLTAGE:  0,
    FaultType.PRED_UNDERVOLTAGE: 0,
    FaultType.PRED_OVERTEMP:     0,
}


@dataclass
class Fault:
    """Single fault event record."""
    fault_type:  FaultType
    timestamp:   float = field(default_factory=time.time)
    cell_id:     Optional[int] = None   # None for pack-level faults
    value:       Optional[float] = None   # Measured value that tripped
    threshold:   Optional[float] = None   # The limit that was exceeded
    severity:    int = 1      # 1 = warning, 2 = critical
    description: str = ""

    @property
    def label(self) -> str:
        return FAULT_LABELS.get(self.fault_type, "UNKNOWN")

    @property
    def is_predictive(self) -> bool:
        return self.fault_type in (
            FaultType.PRED_OVERVOLTAGE,
            FaultType.PRED_UNDERVOLTAGE,
            FaultType.PRED_OVERTEMP,
        )

    def __str__(self) -> str:
        loc = f"cell {self.cell_id}" if self.cell_id is not None else "pack"
        val = f"{self.value:.3f}" if self.value is not None else "—"
        thr = f"{self.threshold:.3f}" if self.threshold is not None else "—"
        sev_map = {0: "PREDICTIVE", 1: "WARNING", 2: "CRITICAL"}
        return (
            f"[{self.label}] {loc}  "
            f"val={val}  limit={thr}  "
            f"sev={sev_map.get(self.severity, 'UNKNOWN')}"
        )


# ---------------------------------------------------------------------------
# Thresholds configuration
# ---------------------------------------------------------------------------

@dataclass
class BmsThresholds:
    # Cell voltage (V)
    overvoltage:      float = 4.20
    undervoltage:     float = 2.50

    # Temperature (°C)
    overtemp_warn:    float = 55.0
    overtemp_critical: float = 60.0

    # Current (A)
    overcurrent_cont: float = 180.0   # continuous
    overcurrent_peak: float = 200.0   # instantaneous trip

    # Cell voltage spread (V) — imbalance warning
    delta_v_warn:     float = 0.20
    delta_v_critical: float = 0.50

    # Isolation resistance (MΩ) — FSAE EV.8.4: 500 Ω/V of pack max voltage
    # 96s × 4.2 V = 403.2 V → threshold ≈ 0.20 MΩ
    iso_resistance_min: float = 0.20  # MΩ

    # How long a fault must persist before tripping (seconds)
    # Prevents nuisance trips from transient sensor noise
    debounce_s: float = 0.05


# ---------------------------------------------------------------------------
# Predictive fault analyser
# ---------------------------------------------------------------------------

class PredictiveAnalyzer:
    """
    Tracks per-series-group voltage and temperature trajectories.
    Uses linear regression on a rolling window to extrapolate values
    forward by `horizon_s` seconds.  If the predicted value breaches a
    threshold while the current value has not, a PREDICTIVE warning is
    emitted — giving the shutdown circuit reaction time.
    """

    HORIZON_S = 10.0   # seconds to look ahead
    MIN_SAMPLES = 15     # need this many samples before predicting
    WINDOW = 30     # samples used for linear fit

    def __init__(self):
        # {series_index: deque of (timestamp, voltage)}
        self._v_buf: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
        # {series_index: deque of (timestamp, temperature)}
        self._t_buf: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
        # Latest predicted values for dashboard display
        self.predicted_voltages: Dict[int, float] = {}
        self.predicted_temps:    Dict[int, float] = {}
        self.voltage_rates:      Dict[int, float] = {}   # V/s
        self.temp_rates:         Dict[int, float] = {}    # °C/s

    def record(self, state: PackState) -> None:
        """Append current readings to the rolling buffers."""
        t = state.timestamp
        for s in range(SERIES_COUNT):
            cells = [c for c in state.cells if c.series_index == s]
            v_avg = float(np.mean([c.voltage for c in cells]))
            t_max = float(max(c.temperature for c in cells))
            self._v_buf[s].append((t, v_avg))
            self._t_buf[s].append((t, t_max))

    def predict(self, th: BmsThresholds) -> List[Fault]:
        """
        Run extrapolation on all series groups and return predictive
        Fault objects for any groups whose trajectory will breach a
        threshold within HORIZON_S seconds.
        """
        warnings: List[Fault] = []
        now = time.time()

        for s in range(SERIES_COUNT):
            # --- Voltage prediction ---
            vbuf = self._v_buf[s]
            if len(vbuf) >= self.MIN_SAMPLES:
                rate, current_v = self._fit_rate(vbuf)
                self.voltage_rates[s] = rate
                pred_v = current_v + rate * self.HORIZON_S
                self.predicted_voltages[s] = pred_v

                # Predict overvoltage (rising toward limit)
                if pred_v > th.overvoltage and current_v <= th.overvoltage:
                    tta = (th.overvoltage - current_v) / \
                        rate if rate > 0 else 99
                    warnings.append(Fault(
                        fault_type=FaultType.PRED_OVERVOLTAGE,
                        timestamp=now,
                        cell_id=s,    # series group index
                        value=current_v,
                        threshold=th.overvoltage,
                        severity=0,
                        description=(
                            f"Group {s} V trending {rate*1000:+.1f} mV/s -> "
                            f"predicted {pred_v:.3f} V in {self.HORIZON_S:.0f}s "
                            f"(ETA {tta:.1f}s)"
                        ),
                    ))

                # Predict undervoltage (falling toward limit)
                if pred_v < th.undervoltage and current_v >= th.undervoltage:
                    tta = (current_v - th.undervoltage) / \
                        (-rate) if rate < 0 else 99
                    warnings.append(Fault(
                        fault_type=FaultType.PRED_UNDERVOLTAGE,
                        timestamp=now,
                        cell_id=s,
                        value=current_v,
                        threshold=th.undervoltage,
                        severity=0,
                        description=(
                            f"Group {s} V trending {rate*1000:+.1f} mV/s -> "
                            f"predicted {pred_v:.3f} V in {self.HORIZON_S:.0f}s "
                            f"(ETA {tta:.1f}s)"
                        ),
                    ))

            # --- Temperature prediction ---
            tbuf = self._t_buf[s]
            if len(tbuf) >= self.MIN_SAMPLES:
                rate, current_t = self._fit_rate(tbuf)
                self.temp_rates[s] = rate
                pred_t = current_t + rate * self.HORIZON_S
                self.predicted_temps[s] = pred_t

                if pred_t > th.overtemp_critical and current_t <= th.overtemp_critical:
                    tta = ((th.overtemp_critical - current_t) / rate
                           if rate > 0 else 99)
                    warnings.append(Fault(
                        fault_type=FaultType.PRED_OVERTEMP,
                        timestamp=now,
                        cell_id=s,
                        value=current_t,
                        threshold=th.overtemp_critical,
                        severity=0,
                        description=(
                            f"Group {s} temp trending {rate:+.2f} C/s -> "
                            f"predicted {pred_t:.1f}C in {self.HORIZON_S:.0f}s "
                            f"(ETA {tta:.1f}s)"
                        ),
                    ))

        return warnings

    def _fit_rate(self, buf: deque) -> Tuple[float, float]:
        """
        Linear regression on the last WINDOW samples.
        Returns (rate_per_second, current_value).
        """
        data = list(buf)[-self.WINDOW:]
        ts = np.array([p[0] for p in data])
        vs = np.array([p[1] for p in data])
        ts -= ts[0]  # normalise to avoid precision issues
        if ts[-1] - ts[0] < 0.01:
            return 0.0, float(vs[-1])
        coeffs = np.polyfit(ts, vs, 1)
        return float(coeffs[0]), float(vs[-1])


# ---------------------------------------------------------------------------
# Fault detector
# ---------------------------------------------------------------------------

class FaultDetector:
    """
    Processes each PackState snapshot and returns a list of active Fault
    objects.  Maintains latch state for the IMD shutdown circuit and AIRs.

    Usage:
        detector = FaultDetector()
        faults   = detector.evaluate(pack_state)
    """

    def __init__(self, thresholds: Optional[BmsThresholds] = None):
        self.th = thresholds or BmsThresholds()

        # Active (currently tripped) faults keyed by (FaultType, cell_id)
        self._active: Dict[Tuple[FaultType, Optional[int]], Fault] = {}

        # Fault onset timestamps for debounce
        self._onset: Dict[Tuple[FaultType, Optional[int]], float] = {}

        # IMD / AIR latch — once triggered, requires manual reset
        self._imd_latched:  bool = False
        self._airs_latched: bool = False   # True = AIRs held open

        # Fault history (all faults ever seen this session)
        self.history: List[Fault] = []

        # Predictive fault analyser
        self.predictor = PredictiveAnalyzer()

        # Active predictive warnings (separate from confirmed faults)
        self._predictive: Dict[Tuple[FaultType, Optional[int]], Fault] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        state: PackState,
        comm_fault: bool = False,
        comm_fault_ic: Optional[int] = None,
    ) -> List[Fault]:
        """
        Evaluate all fault conditions against the current PackState.
        Returns the list of *newly triggered* faults this cycle.
        Updates state.imd_triggered and state.airs_closed in-place.

        Parameters
        ----------
        comm_fault:    True if LTC6813 SPI communication failed this cycle
        comm_fault_ic: which IC in the daisy chain faulted (0-5), or None
        """
        new_faults: List[Fault] = []
        now = state.timestamp

        candidate_faults: List[Fault] = []

        # ---- SPI communication fault ----------------------------------
        if comm_fault:
            candidate_faults.append(Fault(
                fault_type=FaultType.COMM_FAULT,
                cell_id=comm_fault_ic,
                value=None,
                threshold=None,
                severity=FAULT_SEVERITY[FaultType.COMM_FAULT],
                description=(
                    f"SPI COMM_FAULT on IC {comm_fault_ic} -- "
                    f"PEC mismatch after max retries"
                ),
            ))

        # ---- Per-cell checks ----------------------------------------
        for cell in state.cells:
            # Overvoltage
            if cell.voltage > self.th.overvoltage:
                candidate_faults.append(Fault(
                    fault_type=FaultType.OVERVOLTAGE,
                    cell_id=cell.cell_id,
                    value=cell.voltage,
                    threshold=self.th.overvoltage,
                    severity=FAULT_SEVERITY[FaultType.OVERVOLTAGE],
                    description=(
                        f"Cell {cell.cell_id} voltage {cell.voltage:.3f} V "
                        f"> {self.th.overvoltage:.2f} V limit"
                    ),
                ))

            # Undervoltage
            if cell.voltage < self.th.undervoltage:
                candidate_faults.append(Fault(
                    fault_type=FaultType.UNDERVOLTAGE,
                    cell_id=cell.cell_id,
                    value=cell.voltage,
                    threshold=self.th.undervoltage,
                    severity=FAULT_SEVERITY[FaultType.UNDERVOLTAGE],
                    description=(
                        f"Cell {cell.cell_id} voltage {cell.voltage:.3f} V "
                        f"< {self.th.undervoltage:.2f} V limit"
                    ),
                ))

            # Overtemperature (warn first, then critical)
            if cell.temperature > self.th.overtemp_critical:
                candidate_faults.append(Fault(
                    fault_type=FaultType.OVERTEMPERATURE,
                    cell_id=cell.cell_id,
                    value=cell.temperature,
                    threshold=self.th.overtemp_critical,
                    severity=2,
                    description=(
                        f"Cell {cell.cell_id} temp {cell.temperature:.1f}C "
                        f"> {self.th.overtemp_critical:.0f}C critical"
                    ),
                ))
            elif cell.temperature > self.th.overtemp_warn:
                candidate_faults.append(Fault(
                    fault_type=FaultType.OVERTEMPERATURE,
                    cell_id=cell.cell_id,
                    value=cell.temperature,
                    threshold=self.th.overtemp_warn,
                    severity=1,
                    description=(
                        f"Cell {cell.cell_id} temp {cell.temperature:.1f}C "
                        f"> {self.th.overtemp_warn:.0f}C warning"
                    ),
                ))

        # ---- Pack-level checks --------------------------------------

        # Overcurrent (instantaneous peak)
        if abs(state.pack_current) > self.th.overcurrent_peak:
            candidate_faults.append(Fault(
                fault_type=FaultType.OVERCURRENT,
                cell_id=None,
                value=state.pack_current,
                threshold=self.th.overcurrent_peak,
                severity=FAULT_SEVERITY[FaultType.OVERCURRENT],
                description=(
                    f"Pack current {state.pack_current:.1f} A "
                    f"> {self.th.overcurrent_peak:.0f} A peak limit"
                ),
            ))
        elif abs(state.pack_current) > self.th.overcurrent_cont:
            candidate_faults.append(Fault(
                fault_type=FaultType.OVERCURRENT,
                cell_id=None,
                value=state.pack_current,
                threshold=self.th.overcurrent_cont,
                severity=1,
                description=(
                    f"Pack current {state.pack_current:.1f} A "
                    f"> {self.th.overcurrent_cont:.0f} A continuous limit"
                ),
            ))

        # Voltage imbalance
        dv = state.delta_voltage
        if dv > self.th.delta_v_critical:
            candidate_faults.append(Fault(
                fault_type=FaultType.VOLTAGE_IMBALANCE,
                cell_id=None,
                value=dv,
                threshold=self.th.delta_v_critical,
                severity=2,
                description=f"Cell dV {dv:.3f} V > {self.th.delta_v_critical:.2f} V critical",
            ))
        elif dv > self.th.delta_v_warn:
            candidate_faults.append(Fault(
                fault_type=FaultType.VOLTAGE_IMBALANCE,
                cell_id=None,
                value=dv,
                threshold=self.th.delta_v_warn,
                severity=1,
                description=f"Cell dV {dv:.3f} V > {self.th.delta_v_warn:.2f} V warning",
            ))

        # Isolation fault — IMD
        iso_min = min(state.iso_resistance_pos, state.iso_resistance_neg)
        if iso_min < self.th.iso_resistance_min:
            candidate_faults.append(Fault(
                fault_type=FaultType.ISOLATION_FAULT,
                cell_id=None,
                value=iso_min,
                threshold=self.th.iso_resistance_min,
                severity=FAULT_SEVERITY[FaultType.ISOLATION_FAULT],
                description=(
                    f"Isolation resistance {iso_min*1000:.1f} kOhm "
                    f"< {self.th.iso_resistance_min*1000:.0f} kOhm minimum"
                ),
            ))

        # ---- Debounce + latch logic ----------------------------------
        current_keys = set()

        for fault in candidate_faults:
            key = (fault.fault_type, fault.cell_id)
            current_keys.add(key)

            if key not in self._onset:
                self._onset[key] = now   # start debounce timer

            elapsed = now - self._onset[key]
            if elapsed >= self.th.debounce_s:
                if key not in self._active:
                    # Newly confirmed fault
                    fault.timestamp = now
                    self._active[key] = fault
                    self.history.append(fault)
                    new_faults.append(fault)
                else:
                    # Update value on existing fault
                    self._active[key].value = fault.value

        # Clear debounce timers for faults that resolved
        for key in list(self._onset.keys()):
            if key not in current_keys:
                del self._onset[key]

        # Clear active faults that resolved
        for key in list(self._active.keys()):
            if key not in current_keys:
                del self._active[key]

        # ---- Predictive warnings ------------------------------------
        self.predictor.record(state)
        pred_warnings = self.predictor.predict(self.th)

        # Track active predictive warnings (keyed like regular faults)
        new_pred_keys = set()
        for pw in pred_warnings:
            key = (pw.fault_type, pw.cell_id)
            new_pred_keys.add(key)
            if key not in self._predictive:
                self._predictive[key] = pw
                self.history.append(pw)
                new_faults.append(pw)
            else:
                self._predictive[key].value = pw.value
                self._predictive[key].description = pw.description

        for key in list(self._predictive.keys()):
            if key not in new_pred_keys:
                del self._predictive[key]

        # ---- IMD / AIR shutdown circuit -----------------------------
        self._update_shutdown_circuit(state)

        return new_faults

    @property
    def active_faults(self) -> List[Fault]:
        """Currently active (un-cleared) faults including predictive warnings."""
        return list(self._active.values()) + list(self._predictive.values())

    @property
    def predictive_warnings(self) -> List[Fault]:
        """Currently active predictive warnings only."""
        return list(self._predictive.values())

    @property
    def has_critical_fault(self) -> bool:
        return any(f.severity == 2 for f in self._active.values())

    @property
    def imd_triggered(self) -> bool:
        return self._imd_latched

    @property
    def airs_open(self) -> bool:
        return self._airs_latched

    def reset(self) -> None:
        """
        Manual reset of IMD latch and AIR latch after fault is cleared.
        In a real car this is the AMS/BMS reset button.
        """
        self._imd_latched = False
        self._airs_latched = False
        self._active.clear()
        self._onset.clear()
        self._predictive.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_shutdown_circuit(self, state: PackState) -> None:
        """
        Mirror of the FSAE shutdown circuit logic:
          1. IMD trips on isolation fault → latches
          2. Any severity-2 fault → opens AIRs (latches)
          3. State is updated on the PackState object in-place
        """
        iso_fault_active = any(
            f.fault_type == FaultType.ISOLATION_FAULT
            for f in self._active.values()
        )

        if iso_fault_active and not self._imd_latched:
            self._imd_latched = True

        if (self._imd_latched or self.has_critical_fault) and not self._airs_latched:
            self._airs_latched = True

        # Propagate back to pack state
        state.imd_triggered = self._imd_latched
        state.airs_closed = not self._airs_latched
