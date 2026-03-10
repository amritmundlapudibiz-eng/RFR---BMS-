"""
main.py — BMS Fault Detector Entry Point
Rutgers Formula Racing BMS Fault Detector

Ties together the simulator, fault detector, cell balancer, SOH estimator,
logger, and dashboard.

Usage:
    python main.py                      # default: auto fault-injection demo
    python main.py --no-demo            # clean run, no injected faults
    python main.py --initial-soc 0.95   # start at 95% SOC
    python main.py --tick 0.1           # 100 ms simulation steps
    python main.py --help

Keyboard controls (while running):
    Q         — quit
    1–5       — change drive mode  (1=idle 2=cruise 3=accel 4=regen 5=charge)
    F         — cycle through fault injection types
    R         — reset all injected faults + IMD/AIR latch
    S         — write full cell snapshot CSV
"""

import argparse
import sys
import threading
import time
from typing import Optional

# Windows-specific: enable ANSI escape codes in the terminal
if sys.platform == "win32":
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

from rich.console import Console

from balancer import CellBalancer, BalancerConfig
from dashboard import Dashboard
from fault_detector import FaultDetector, BmsThresholds
from logger import DataLogger
from ltc6813_interface import Ltc6813Interface, Ltc6813Config
from simulator import PackSimulator, SERIES_COUNT, PARALLEL_COUNT
from soh import SohEstimator, SohConfig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rutgers Formula Racing — BMS Fault Detector Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--initial-soc",  type=float, default=0.80,
                   help="Initial pack SOC (0–1)")
    p.add_argument("--ambient-temp", type=float, default=25.0,
                   help="Ambient temperature (°C)")
    p.add_argument("--tick",         type=float, default=0.10,
                   help="Simulation time step (s)")
    p.add_argument("--no-demo",      action="store_true",
                   help="Disable automatic fault injection demo sequence")
    p.add_argument("--log-dir",      type=str,   default="logs",
                   help="Directory for CSV log files")
    p.add_argument("--seed",         type=int,   default=42,
                   help="RNG seed for reproducible simulation")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Keyboard input (non-blocking, cross-platform)
# ---------------------------------------------------------------------------

class KeyReader:
    """Non-blocking single-character keyboard reader."""

    def __init__(self):
        self._key: Optional[str] = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self):
        try:
            if sys.platform == "win32":
                import msvcrt
                while True:
                    if msvcrt.kbhit():
                        ch = msvcrt.getwch()
                        with self._lock:
                            self._key = ch.lower()
                    time.sleep(0.02)
            else:
                import tty
                import termios
                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                try:
                    tty.setraw(fd)
                    while True:
                        ch = sys.stdin.read(1)
                        with self._lock:
                            self._key = ch.lower()
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass  # terminal not attached (redirect / CI)

    def get(self) -> Optional[str]:
        with self._lock:
            ch = self._key
            self._key = None
            return ch


# ---------------------------------------------------------------------------
# Demo fault schedule
# ---------------------------------------------------------------------------

def schedule_demo_faults(sim: PackSimulator) -> None:
    """
    Pre-schedule a sequence of injected faults for demonstration.
    Times are relative to simulation start (seconds).
    """
    # t=8s  — single cell overvoltage (regen over-charge scenario)
    sim.inject_fault("overvoltage",  cell_index=47,  delay_s=8.0)

    # t=15s — isolation fault (coolant leak onto HV bus)
    sim.inject_fault("isolation",    cell_index=None, delay_s=15.0)

    # t=22s — cell undervoltage (deeply discharged cell)
    sim.inject_fault("undervoltage", cell_index=112, delay_s=22.0)

    # t=30s — overtemperature (cooling failure)
    sim.inject_fault("overtemp",     cell_index=72,  delay_s=30.0)

    # t=38s — overcurrent spike (motor controller fault)
    sim.inject_fault("overcurrent",  cell_index=None, delay_s=38.0)


# ---------------------------------------------------------------------------
# Drive mode key bindings
# ---------------------------------------------------------------------------

MODE_KEYS = {
    "1": "idle",
    "2": "cruising",
    "3": "accelerating",
    "4": "braking",
    "5": "charging",
}

FAULT_CYCLE = ["overvoltage", "undervoltage",
               "overtemp", "overcurrent", "isolation"]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    console = Console()
    console.print(
        "[bold cyan]Rutgers Formula Racing — BMS Fault Detector[/bold cyan]")
    console.print("Initialising pack simulator …")

    # ---- Instantiate all subsystems ---------------------------------
    sim = PackSimulator(
        initial_soc=args.initial_soc,
        ambient_temp=args.ambient_temp,
        seed=args.seed,
    )

    detector = FaultDetector(BmsThresholds())
    balancer = CellBalancer(BalancerConfig())
    soh_est = SohEstimator(SohConfig())
    ltc6813 = Ltc6813Interface(Ltc6813Config(), seed=args.seed)
    keys = KeyReader()

    console.print(
        f"[dim]LTC6813 SPI interface initialized -- "
        f"6 ICs, PEC error rate {ltc6813.cfg.pec_error_rate*100:.0f}%, "
        f"ADC {ltc6813.cfg.adc_latency_us:.0f}us latency[/dim]"
    )

    # Initialize SOH estimator with current cell states
    soh_est.initialize(sim.state.cells)
    console.print(
        f"[dim]SOH initialized — min {soh_est.min_soh:.1f}%, "
        f"avg {soh_est.avg_soh:.1f}%, "
        f"{len(soh_est.degraded_cells)} degraded cells[/dim]"
    )

    if not args.no_demo:
        schedule_demo_faults(sim)
        sim.set_load("cruising")
        console.print(
            "[green]Demo fault injection sequence scheduled.[/green]")

    fault_cycle_idx = 0
    drive_mode = "cruising" if not args.no_demo else "idle"
    sim.set_load(drive_mode)

    start_wall = time.time()

    with DataLogger(log_dir=args.log_dir) as logger:
        with Dashboard(console=console) as dash:

            console.print(
                "[dim]Starting live dashboard … press Q to quit[/dim]")
            time.sleep(0.5)

            try:
                while True:
                    dt = args.tick

                    # ---- Physics step --------------------------------
                    state = sim.step(dt=dt)
                    elapsed = time.time() - start_wall

                    # ---- LTC6813 SPI acquisition ---------------------
                    # Reads ideal physics, quantizes through ADC, applies
                    # PEC15 CRC, injects SPI faults, converts NTC temps
                    adc_v, adc_t, comm_ok = ltc6813.read_all(state)

                    # Apply ADC-quantized readings back to pack state
                    # so downstream (fault detector, balancer) sees what
                    # the real BMS hardware would actually measure.
                    # Skip cells with injected faults — their physics are
                    # held by the simulator and must reach fault_detector.
                    for s in range(SERIES_COUNT):
                        for c in state.cells:
                            if c.series_index == s and not c.injected_fault:
                                if s < len(adc_v):
                                    c.voltage = adc_v[s]
                                if s < len(adc_t):
                                    c.temperature = adc_t[s]

                    # ---- SOH estimation ------------------------------
                    soh_est.update(state, dt)

                    # ---- Cell balancing ------------------------------
                    bleed_plan = balancer.evaluate(state, dt)
                    if bleed_plan:
                        balancer.apply_bleed(state, bleed_plan, dt)
                        # Write discharge config to LTC6813 WRCFGA
                        dcc_map = {s: True for s in bleed_plan}
                        ltc6813.write_discharge_config(dcc_map)

                    # Log completed balancing events
                    completed_bal = balancer.get_new_completed()
                    if completed_bal:
                        logger.log_balancing_events(completed_bal)

                    # ---- Fault detection (+ predictive) ---------------
                    comm_fault_ic = (
                        ltc6813._last_comm_fault_ic
                        if not comm_ok else None
                    )
                    new_faults = detector.evaluate(
                        state,
                        comm_fault=not comm_ok,
                        comm_fault_ic=comm_fault_ic,
                    )
                    active = detector.active_faults

                    # ---- Logging ------------------------------------
                    logger.log_faults(new_faults)
                    logger.log_pack_state(
                        state,
                        active_faults=len(active),
                        min_soh=soh_est.min_soh,
                        avg_soh=soh_est.avg_soh,
                        balancing=balancer.is_balancing,
                    )

                    # ---- Dashboard render ---------------------------
                    dash.update(
                        state, active,
                        mode=drive_mode,
                        elapsed_s=elapsed,
                        balancer=balancer,
                        soh_estimator=soh_est,
                        spi_iface=ltc6813,
                    )

                    # ---- Keyboard input -----------------------------
                    key = keys.get()
                    if key == "q":
                        break

                    elif key in MODE_KEYS:
                        drive_mode = MODE_KEYS[key]
                        sim.set_load(drive_mode)

                    elif key == "f":
                        fault_type = FAULT_CYCLE[fault_cycle_idx % len(
                            FAULT_CYCLE)]
                        fault_cycle_idx += 1
                        sim.inject_fault(fault_type, delay_s=0.0)

                    elif key == "r":
                        sim.clear_faults()
                        detector.reset()

                    elif key == "s":
                        path = logger.log_cell_snapshot(state, label="manual")

                    # ---- Pace the loop ------------------------------
                    time.sleep(max(0.0, args.tick - 0.005))

            except KeyboardInterrupt:
                pass

    # ---- Shutdown summary -------------------------------------------
    console.print()
    console.print("[bold cyan]— Simulation ended —[/bold cyan]")
    console.print(logger.summary())
    spi_stats = ltc6813.get_stats()
    console.print(
        f"Fault history: [bold]{len(detector.history)}[/bold] total events\n"
        f"  Predictive warnings: "
        f"[bold magenta]{sum(1 for f in detector.history if f.is_predictive)}[/bold magenta]\n"
        f"  SOH: min {soh_est.min_soh:.1f}%, avg {soh_est.avg_soh:.1f}%, "
        f"{len(soh_est.degraded_cells)} degraded\n"
        f"  Balancing: {balancer.total_events} events, "
        f"{balancer.total_energy_mwh:.1f} mWh dissipated\n"
        f"  SPI: {spi_stats.total_frames:,} frames, "
        f"{spi_stats.pec_mismatches} PEC errors, "
        f"{spi_stats.comm_faults} COMM_FAULTs\n"
        f"  Log dir: [link]{args.log_dir}[/link]"
    )

    if detector.history:
        console.print()
        console.print("[bold]Last 20 fault events:[/bold]")
        for f in detector.history[-20:]:
            sev_map = {0: "[magenta]PREDICT [/magenta]",
                       1: "[yellow]WARNING [/yellow]",
                       2: "[bold red]CRITICAL[/bold red]"}
            sev = sev_map.get(f.severity, "[dim]UNKNOWN[/dim]")
            console.print(f"  {sev}  {f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
