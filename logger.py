"""
logger.py — BMS Data & Fault Logger
Rutgers Formula Racing BMS Fault Detector

Writes CSVs:
  bms_faults_<ts>.csv      — one row per fault event (including predictive)
  bms_telemetry_<ts>.csv   — periodic pack-level snapshots
  bms_balancing_<ts>.csv   — cell balancing events

All timestamps are Unix epoch seconds (float) for easy post-processing.
"""

from __future__ import annotations

import csv
import os
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from fault_detector import Fault
from simulator import PackState

if TYPE_CHECKING:
    from balancer import BalancingEvent
    from soh import SohEstimator


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

_FAULT_COLUMNS = [
    "timestamp",
    "datetime",
    "fault_type",
    "severity",
    "cell_id",
    "value",
    "threshold",
    "description",
]

_TELEMETRY_COLUMNS = [
    "timestamp",
    "datetime",
    "pack_voltage_V",
    "pack_current_A",
    "pack_soc_pct",
    "min_cell_voltage_V",
    "max_cell_voltage_V",
    "avg_cell_voltage_V",
    "delta_voltage_V",
    "min_cell_temp_C",
    "max_cell_temp_C",
    "avg_cell_temp_C",
    "iso_resistance_pos_Mohm",
    "iso_resistance_neg_Mohm",
    "airs_closed",
    "imd_triggered",
    "active_fault_count",
    "min_soh_pct",
    "avg_soh_pct",
    "balancing_active",
]

_BALANCING_COLUMNS = [
    "timestamp",
    "datetime",
    "series_index",
    "start_voltage_V",
    "target_voltage_V",
    "voltage_delta_mV",
    "bleed_current_mA",
    "duration_s",
    "end_voltage_V",
]


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class DataLogger:
    """
    Thread-safe CSV logger for BMS fault events, telemetry, and balancing.

    Parameters
    ----------
    log_dir:            directory for output files (created if absent)
    telemetry_interval: seconds between telemetry snapshots (0 = every step)
    session_label:      prefix for filenames; defaults to ISO timestamp
    """

    def __init__(
        self,
        log_dir:            str = "logs",
        telemetry_interval: float = 1.0,
        session_label:      Optional[str] = None,
    ):
        self._lock = threading.Lock()
        self._closed = False

        os.makedirs(log_dir, exist_ok=True)
        self._log_dir = log_dir

        label = session_label or datetime.now().strftime("%Y%m%d_%H%M%S")

        fault_path = os.path.join(log_dir, f"bms_faults_{label}.csv")
        telemetry_path = os.path.join(log_dir, f"bms_telemetry_{label}.csv")
        balancing_path = os.path.join(log_dir, f"bms_balancing_{label}.csv")

        self._fault_fh = open(fault_path,     "w",
                              newline="", buffering=1, encoding="utf-8")
        self._telemetry_fh = open(
            telemetry_path, "w", newline="", buffering=1, encoding="utf-8")
        self._balancing_fh = open(
            balancing_path,  "w", newline="", buffering=1, encoding="utf-8")

        self._fault_writer = csv.DictWriter(
            self._fault_fh,     fieldnames=_FAULT_COLUMNS)
        self._telemetry_writer = csv.DictWriter(
            self._telemetry_fh, fieldnames=_TELEMETRY_COLUMNS)
        self._balancing_writer = csv.DictWriter(
            self._balancing_fh, fieldnames=_BALANCING_COLUMNS)

        self._fault_writer.writeheader()
        self._telemetry_writer.writeheader()
        self._balancing_writer.writeheader()

        self.telemetry_interval = telemetry_interval
        self._last_telemetry_time = 0.0

        self.fault_path = fault_path
        self.telemetry_path = telemetry_path
        self.balancing_path = balancing_path

        self.fault_count = 0
        self.telemetry_count = 0
        self.balancing_count = 0

        print(
            f"[Logger] Fault log   -> {fault_path}\n"
            f"[Logger] Telemetry  -> {telemetry_path}\n"
            f"[Logger] Balancing  -> {balancing_path}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_faults(self, faults: List[Fault]) -> None:
        """Log a list of newly triggered Fault objects (including predictive)."""
        if not faults or self._closed:
            return
        with self._lock:
            for fault in faults:
                sev_map = {0: "PREDICTIVE", 1: "WARNING", 2: "CRITICAL"}
                self._fault_writer.writerow({
                    "timestamp":   fault.timestamp,
                    "datetime":    _ts_to_dt(fault.timestamp),
                    "fault_type":  fault.label,
                    "severity":    sev_map.get(fault.severity, "UNKNOWN"),
                    "cell_id":     fault.cell_id if fault.cell_id is not None else "",
                    "value":       f"{fault.value:.4f}" if fault.value is not None else "",
                    "threshold":   f"{fault.threshold:.4f}" if fault.threshold is not None else "",
                    "description": fault.description,
                })
                self.fault_count += 1

    def log_pack_state(
        self,
        state:         PackState,
        active_faults: int = 0,
        min_soh:       float = 100.0,
        avg_soh:       float = 100.0,
        balancing:     bool = False,
    ) -> bool:
        """
        Log a telemetry row if the interval has elapsed.
        Returns True if a row was written.
        """
        if self._closed:
            return False
        now = state.timestamp
        if now - self._last_telemetry_time < self.telemetry_interval:
            return False

        with self._lock:
            self._telemetry_writer.writerow({
                "timestamp":              now,
                "datetime":               _ts_to_dt(now),
                "pack_voltage_V":         f"{state.pack_voltage:.2f}",
                "pack_current_A":         f"{state.pack_current:.2f}",
                "pack_soc_pct":           f"{state.pack_soc * 100:.1f}",
                "min_cell_voltage_V":     f"{state.min_cell_voltage:.4f}",
                "max_cell_voltage_V":     f"{state.max_cell_voltage:.4f}",
                "avg_cell_voltage_V":     f"{state.avg_cell_voltage:.4f}",
                "delta_voltage_V":        f"{state.delta_voltage:.4f}",
                "min_cell_temp_C":        f"{state.min_cell_temp:.2f}",
                "max_cell_temp_C":        f"{state.max_cell_temp:.2f}",
                "avg_cell_temp_C":        f"{state.avg_cell_temp:.2f}",
                "iso_resistance_pos_Mohm": f"{state.iso_resistance_pos:.4f}",
                "iso_resistance_neg_Mohm": f"{state.iso_resistance_neg:.4f}",
                "airs_closed":            state.airs_closed,
                "imd_triggered":          state.imd_triggered,
                "active_fault_count":     active_faults,
                "min_soh_pct":            f"{min_soh:.1f}",
                "avg_soh_pct":            f"{avg_soh:.1f}",
                "balancing_active":       balancing,
            })
            self._last_telemetry_time = now
            self.telemetry_count += 1
            return True

    def log_balancing_events(self, events: list) -> None:
        """Log completed balancing events."""
        if not events or self._closed:
            return
        with self._lock:
            for ev in events:
                now = time.time()
                self._balancing_writer.writerow({
                    "timestamp":       now,
                    "datetime":        _ts_to_dt(now),
                    "series_index":    ev.series_index,
                    "start_voltage_V": f"{ev.start_voltage:.4f}",
                    "target_voltage_V": f"{ev.target_voltage:.4f}",
                    "voltage_delta_mV": f"{ev.voltage_delta * 1000:.1f}",
                    "bleed_current_mA": f"{ev.bleed_current_ma:.1f}",
                    "duration_s":      f"{ev.duration_s:.2f}",
                    "end_voltage_V":   f"{ev.end_voltage:.4f}",
                })
                self.balancing_count += 1

    def log_cell_snapshot(self, state: PackState, label: str = "snapshot") -> str:
        """
        Write a one-time full cell dump to a separate CSV file.
        Returns the path of the file written.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self._log_dir, f"bms_cells_{label}_{ts}.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "cell_id", "series_index", "parallel_index",
                "voltage_V", "temperature_C", "current_A",
                "soc_pct", "soh_pct", "internal_resistance_ohm",
                "injected_fault",
            ])
            writer.writeheader()
            for c in state.cells:
                writer.writerow({
                    "cell_id":                c.cell_id,
                    "series_index":           c.series_index,
                    "parallel_index":         c.parallel_index,
                    "voltage_V":              f"{c.voltage:.4f}",
                    "temperature_C":          f"{c.temperature:.2f}",
                    "current_A":              f"{c.current:.3f}",
                    "soc_pct":                f"{c.soc * 100:.2f}",
                    "soh_pct":                f"{c.soh:.1f}",
                    "internal_resistance_ohm": f"{c.internal_resistance:.5f}",
                    "injected_fault":         c.injected_fault or "",
                })
        return path

    def close(self) -> None:
        """Flush and close all file handles."""
        with self._lock:
            if not self._closed:
                for fh in (self._fault_fh, self._telemetry_fh, self._balancing_fh):
                    fh.flush()
                    fh.close()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def summary(self) -> str:
        return (
            f"Logger closed — "
            f"{self.fault_count} fault events, "
            f"{self.telemetry_count} telemetry rows, "
            f"{self.balancing_count} balancing events written."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts_to_dt(ts: float) -> str:
    """Convert Unix timestamp to ISO 8601 string."""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
