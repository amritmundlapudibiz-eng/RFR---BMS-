"""
dashboard.py — Live BMS Terminal Dashboard
Rutgers Formula Racing BMS Fault Detector

Renders a live Rich-based dashboard showing:
  • Pack summary (voltage, current, SOC, temps, IMD/AIR status)
  • 96-cell voltage grid with colour-coded health
  • 96-cell temperature grid
  • SOH summary + degraded-cell list
  • Cell balancing status panel
  • Active fault list (including predictive warnings)
  • IMD / shutdown circuit status

Designed to run inside a Rich Live context.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from fault_detector import Fault, FaultType
from simulator import PackState, SERIES_COUNT, PARALLEL_COUNT

if TYPE_CHECKING:
    from balancer import CellBalancer
    from ltc6813_interface import Ltc6813Interface, SpiStats
    from soh import SohEstimator

# ---------------------------------------------------------------------------
# Colour thresholds (match fault_detector thresholds)
# ---------------------------------------------------------------------------
_V_OVER = 4.20
_V_WARN_H = 4.10
_V_WARN_L = 2.80
_V_UNDER = 2.50
_T_CRIT = 60.0
_T_WARN = 55.0
_T_NORM = 45.0


def _voltage_style(v: float) -> str:
    if v >= _V_OVER:
        return "bold red"
    if v >= _V_WARN_H:
        return "yellow"
    if v <= _V_UNDER:
        return "bold red"
    if v <= _V_WARN_L:
        return "yellow"
    return "green"


def _temp_style(t: float) -> str:
    if t >= _T_CRIT:
        return "bold red"
    if t >= _T_WARN:
        return "yellow"
    if t >= _T_NORM:
        return "cyan"
    return "green"


def _soh_style(s: float) -> str:
    if s < 70:
        return "bold red"
    if s < 80:
        return "red"
    if s < 85:
        return "yellow"
    if s < 90:
        return "cyan"
    return "green"


def _current_style(i: float, crit: float = 200.0, warn: float = 150.0) -> str:
    a = abs(i)
    if a >= crit:
        return "bold red"
    if a >= warn:
        return "yellow"
    return "bright_white"


def _soc_style(s: float) -> str:
    if s < 0.10:
        return "bold red"
    if s < 0.20:
        return "yellow"
    if s < 0.30:
        return "cyan"
    return "green"


# ---------------------------------------------------------------------------
# Dashboard renderer
# ---------------------------------------------------------------------------

class Dashboard:
    """
    Stateless renderer — call update() each tick with all the data.
    """

    def __init__(self, console: Optional[Console] = None, refresh_per_second: int = 4):
        self.console = console or Console()
        self.refresh_per_second = refresh_per_second
        self._live: Optional[Live] = None
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> Dashboard:
        self._live = Live(
            console=self.console,
            refresh_per_second=self.refresh_per_second,
            screen=True,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args):
        if self._live:
            self._live.__exit__(*args)

    def update(
        self,
        state:         PackState,
        active_faults: List[Fault],
        mode:          str = "idle",
        elapsed_s:     Optional[float] = None,
        balancer:      Optional[CellBalancer] = None,
        soh_estimator: Optional[SohEstimator] = None,
        spi_iface:     Optional[Ltc6813Interface] = None,
    ) -> None:
        """Push a new frame to the Live display."""
        if self._live:
            layout = self.build_layout(
                state, active_faults, mode, elapsed_s,
                balancer, soh_estimator, spi_iface,
            )
            self._live.update(layout)

    # ------------------------------------------------------------------
    # Layout builder
    # ------------------------------------------------------------------

    def build_layout(
        self,
        state:         PackState,
        active_faults: List[Fault],
        mode:          str = "idle",
        elapsed_s:     Optional[float] = None,
        balancer:      Optional[CellBalancer] = None,
        soh_estimator: Optional[SohEstimator] = None,
        spi_iface:     Optional[Ltc6813Interface] = None,
    ):
        elapsed = elapsed_s if elapsed_s is not None else time.time() - self._start_time

        layout = Layout()
        layout.split_column(
            Layout(name="header",  size=3),
            Layout(name="body",    ratio=1),
            Layout(name="footer",  size=3),
        )

        layout["body"].split_row(
            Layout(name="left",   ratio=3),
            Layout(name="right",  ratio=2),
        )

        layout["left"].split_column(
            Layout(name="summary",  size=10),
            Layout(name="voltages", ratio=1),
            Layout(name="temps",    ratio=1),
        )

        layout["right"].split_column(
            Layout(name="status_row", size=12),
            Layout(name="faults",     ratio=1),
            Layout(name="bottom_row", size=9),
        )

        # Top status: SOH | Balancing | SPI side by side
        layout["status_row"].split_row(
            Layout(name="soh_panel", ratio=1),
            Layout(name="bal_panel", ratio=1),
            Layout(name="spi_panel", ratio=1),
        )

        # Bottom: IMD panel (full width)
        layout["bottom_row"].update(self._render_imd_panel(state))

        # ---- Render sections ----
        layout["header"].update(self._render_header(elapsed))
        layout["summary"].update(
            self._render_summary(state, mode, soh_estimator))
        layout["voltages"].update(self._render_voltage_grid(state))
        layout["temps"].update(self._render_temp_grid(state))
        layout["soh_panel"].update(self._render_soh_panel(soh_estimator))
        layout["bal_panel"].update(self._render_balancing_panel(balancer))
        layout["spi_panel"].update(self._render_spi_panel(spi_iface))
        layout["faults"].update(self._render_fault_list(active_faults))
        layout["footer"].update(self._render_footer())

        return layout

    # ------------------------------------------------------------------
    # Renderable sections
    # ------------------------------------------------------------------

    def _render_header(self, elapsed_s: float) -> Panel:
        mins, secs = divmod(int(elapsed_s), 60)
        now_str = datetime.now().strftime("%H:%M:%S.%f")[:-4]
        title = Text.assemble(
            ("  RUTGERS FORMULA RACING — BMS FAULT DETECTOR  ",
             "bold white on dark_red"),
            (f"   {now_str}  +{mins:02d}:{secs:02d}  ", "dim"),
        )
        return Panel(title, box=box.HEAVY, style="on grey11")

    def _render_summary(
        self, state: PackState, mode: str,
        soh_est: Optional[SohEstimator],
    ) -> Panel:
        g = Table.grid(expand=True, padding=(0, 2))
        g.add_column(justify="left",  ratio=1)
        g.add_column(justify="right", ratio=1)
        g.add_column(justify="left",  ratio=1)
        g.add_column(justify="right", ratio=1)
        g.add_column(justify="left",  ratio=1)
        g.add_column(justify="right", ratio=1)

        v_style = _current_style(state.pack_voltage, crit=420, warn=395)
        i_style = _current_style(state.pack_current)
        s_style = _soc_style(state.pack_soc)
        t_style = _temp_style(state.max_cell_temp)

        soh_min = soh_est.min_soh if soh_est else 100.0

        g.add_row(
            "[dim]Pack Voltage[/]",   Text(
                f"{state.pack_voltage:>7.2f} V",  style=v_style),
            "[dim]Pack Current[/]",   Text(
                f"{state.pack_current:>7.1f} A",  style=i_style),
            "[dim]Pack SOC[/]",        Text(
                f"{state.pack_soc*100:>6.1f} %",  style=s_style),
        )
        g.add_row(
            "[dim]Min Cell V[/]",      Text(
                f"{state.min_cell_voltage:>7.4f} V", style=_voltage_style(state.min_cell_voltage)),
            "[dim]Max Cell V[/]",      Text(
                f"{state.max_cell_voltage:>7.4f} V", style=_voltage_style(state.max_cell_voltage)),
            "[dim]ΔV  [/]",            Text(f"{state.delta_voltage*1000:>6.1f} mV",
                                            style="yellow" if state.delta_voltage > 0.20 else "green"),
        )
        g.add_row(
            "[dim]Min Temp[/]",        Text(
                f"{state.min_cell_temp:>7.1f} °C", style="green"),
            "[dim]Max Temp[/]",        Text(
                f"{state.max_cell_temp:>7.1f} °C", style=t_style),
            "[dim]Mode[/]",            Text(f"{mode.upper():>10s}",
                                            style="bold cyan"),
        )
        g.add_row(
            "[dim]Min SOH[/]",         Text(f"{soh_min:>6.1f} %",
                                            style=_soh_style(soh_min)),
            "[dim]Avg SOH[/]",         Text(f"{(soh_est.avg_soh if soh_est else 100):>6.1f} %",
                                            style=_soh_style(soh_est.avg_soh if soh_est else 100)),
            "",                        Text(""),
        )

        return Panel(g, title="[bold]PACK SUMMARY[/bold]", box=box.ROUNDED, style="on grey11")

    def _render_voltage_grid(self, state: PackState) -> Panel:
        """96 cells as a compact grid (16 columns × 6 rows)."""
        COLS = 16
        tbl = Table(box=None, padding=(0, 0), show_header=False, expand=True)
        for _ in range(COLS):
            tbl.add_column(justify="center", min_width=7)

        series_v = []
        for s in range(SERIES_COUNT):
            grp = [c for c in state.cells if c.series_index == s]
            series_v.append(sum(c.voltage for c in grp) / len(grp))

        row: List[Text] = []
        for v in series_v:
            row.append(
                Text(f"{v:.3f}", style=_voltage_style(v), justify="center"))
            if len(row) == COLS:
                tbl.add_row(*row)
                row = []
        if row:
            while len(row) < COLS:
                row.append(Text(""))
            tbl.add_row(*row)

        return Panel(tbl, title="[bold]CELL VOLTAGES (V) — 96 series groups[/bold]",
                     box=box.ROUNDED, style="on grey11")

    def _render_temp_grid(self, state: PackState) -> Panel:
        """96 cells temperature grid (16 cols × 6 rows)."""
        COLS = 16
        tbl = Table(box=None, padding=(0, 0), show_header=False, expand=True)
        for _ in range(COLS):
            tbl.add_column(justify="center", min_width=6)

        series_t = []
        for s in range(SERIES_COUNT):
            grp = [c for c in state.cells if c.series_index == s]
            series_t.append(max(c.temperature for c in grp))

        row: List[Text] = []
        for t in series_t:
            row.append(
                Text(f"{t:.1f}", style=_temp_style(t), justify="center"))
            if len(row) == COLS:
                tbl.add_row(*row)
                row = []
        if row:
            while len(row) < COLS:
                row.append(Text(""))
            tbl.add_row(*row)

        return Panel(tbl, title="[bold]CELL TEMPS (°C) — max per series group[/bold]",
                     box=box.ROUNDED, style="on grey11")

    # ------------------------------------------------------------------
    # NEW: SOH panel
    # ------------------------------------------------------------------

    def _render_soh_panel(self, soh_est: Optional[SohEstimator]) -> Panel:
        """Compact SOH summary with degraded cell list."""
        g = Table.grid(expand=True, padding=(0, 1))
        g.add_column(justify="left")
        g.add_column(justify="right")

        if soh_est is None:
            g.add_row("[dim]SOH estimator not active[/dim]", "")
            return Panel(g, title="[bold]SOH[/bold]", box=box.ROUNDED, style="on grey11")

        min_soh = soh_est.min_soh
        avg_soh = soh_est.avg_soh
        degraded = soh_est.degraded_cells
        max_cyc = soh_est.max_cycles

        g.add_row("[dim]Min SOH[/dim]",
                  Text(f"{min_soh:.1f}%",  style=_soh_style(min_soh)))
        g.add_row("[dim]Avg SOH[/dim]",
                  Text(f"{avg_soh:.1f}%",  style=_soh_style(avg_soh)))
        g.add_row("[dim]Max Cycles[/dim]",
                  Text(f"{max_cyc:.0f}",   style="dim"))
        g.add_row("[dim]Degraded[/dim]",      Text(
            f"{len(degraded)} cells",
            style="bold red" if degraded else "green",
        ))

        # Show worst cells
        if degraded:
            worst = sorted(degraded, key=lambda d: d.soh)[:5]
            for d in worst:
                g.add_row(
                    Text(f"  Cell {d.cell_id}", style="dim"),
                    Text(f"{d.soh:.1f}%", style=_soh_style(d.soh)),
                )

        return Panel(g, title="[bold]STATE OF HEALTH[/bold]",
                     box=box.ROUNDED, style="on grey11")

    # ------------------------------------------------------------------
    # NEW: Balancing panel
    # ------------------------------------------------------------------

    def _render_balancing_panel(self, balancer: Optional[CellBalancer]) -> Panel:
        """Cell balancing status and recent events."""
        g = Table.grid(expand=True, padding=(0, 1))
        g.add_column(justify="left")
        g.add_column(justify="right")

        if balancer is None:
            g.add_row("[dim]Balancer not active[/dim]", "")
            return Panel(g, title="[bold]BALANCING[/bold]", box=box.ROUNDED, style="on grey11")

        is_active = balancer.is_balancing
        status_style = "bold yellow" if is_active else "green"
        status_text = f"ACTIVE ({balancer.active_count})" if is_active else "IDLE"

        g.add_row("[dim]Status[/dim]",
                  Text(status_text, style=status_style))
        g.add_row("[dim]Total Events[/dim]",
                  Text(str(balancer.total_events), style="dim"))
        g.add_row("[dim]Energy Diss.[/dim]",
                  Text(f"{balancer.total_energy_mwh:.1f} mWh", style="dim"))

        # Show currently bleeding cells
        if is_active:
            for s_idx, ev in sorted(balancer.active_bleeds.items())[:5]:
                g.add_row(
                    Text(f"  Grp {s_idx}", style="yellow"),
                    Text(f"Δ{ev.voltage_delta*1000:.0f}mV {ev.bleed_current_ma:.0f}mA",
                         style="yellow"),
                )

        return Panel(g, title="[bold]CELL BALANCING[/bold]",
                     box=box.ROUNDED, style="on grey11")

    # ------------------------------------------------------------------
    # SPI COMMS panel (LTC6813 interface stats)
    # ------------------------------------------------------------------

    def _render_spi_panel(self, spi_iface: Optional[Ltc6813Interface]) -> Panel:
        """LTC6813 SPI bus statistics."""
        g = Table.grid(expand=True, padding=(0, 1))
        g.add_column(justify="left")
        g.add_column(justify="right")

        if spi_iface is None:
            g.add_row("[dim]SPI not active[/dim]", "")
            return Panel(g, title="[bold]SPI COMMS[/bold]", box=box.ROUNDED, style="on grey11")

        stats = spi_iface.get_stats()

        pec_style = "bold red" if stats.pec_mismatches > 0 else "green"
        cf_style = "bold red" if stats.comm_faults > 0 else "green"

        pec_rate = (
            f"{stats.pec_mismatches / stats.total_frames * 100:.2f}%"
            if stats.total_frames > 0 else "0.00%"
        )

        g.add_row("[dim]Frames[/dim]",
                  Text(f"{stats.total_frames:,}", style="dim"))
        g.add_row("[dim]PEC Err[/dim]",
                  Text(f"{stats.pec_mismatches} ({pec_rate})", style=pec_style))
        g.add_row("[dim]Retries[/dim]",
                  Text(f"{stats.retries}", style="yellow" if stats.retries else "dim"))
        g.add_row("[dim]COMM_FAULT[/dim]",
                  Text(f"{stats.comm_faults}", style=cf_style))
        g.add_row("[dim]ADC Lat.[/dim]",
                  Text(f"{stats.avg_latency_us:.0f} us", style="dim"))
        g.add_row("[dim]Bytes TX[/dim]",
                  Text(f"{stats.total_bytes_tx:,}", style="dim"))
        g.add_row("[dim]Bytes RX[/dim]",
                  Text(f"{stats.total_bytes_rx:,}", style="dim"))

        border = "on grey11"
        if stats.comm_faults > 0:
            border = "on dark_red"
        return Panel(g, title="[bold]SPI COMMS[/bold]",
                     box=box.ROUNDED, style=border)

    # ------------------------------------------------------------------
    # Fault list (updated with predictive warnings)
    # ------------------------------------------------------------------

    def _render_fault_list(self, active_faults: List[Fault]) -> Panel:
        tbl = Table(
            box=box.SIMPLE,
            show_header=True,
            expand=True,
            style="on grey11",
            header_style="bold white",
        )
        tbl.add_column("Time",       width=10, style="dim")
        tbl.add_column("Type",       width=14)
        tbl.add_column("Sev",        width=11)
        tbl.add_column("Cell",       width=5,  justify="right")
        tbl.add_column("Value",      width=10, justify="right")
        tbl.add_column("Description", ratio=1)

        shown = sorted(active_faults, key=lambda f: f.timestamp,
                       reverse=True)[:20]
        if not shown:
            tbl.add_row("—", "[dim]no active faults[/dim]", "", "", "", "")
        else:
            for fault in shown:
                sev_map = {0: ("PREDICTIVE", "bold magenta"),
                           1: ("WARNING",    "yellow"),
                           2: ("CRITICAL",   "bold red")}
                sev_label, sev_style = sev_map.get(
                    fault.severity, ("UNKNOWN", "dim"))
                val_str = f"{fault.value:.3f}" if fault.value is not None else "—"
                cell_str = str(
                    fault.cell_id) if fault.cell_id is not None else "pack"
                ts_str = datetime.fromtimestamp(
                    fault.timestamp).strftime("%H:%M:%S")

                # Predictive warnings get distinct styling
                type_style = "bold magenta" if fault.is_predictive else sev_style

                tbl.add_row(
                    ts_str,
                    Text(fault.label,     style=type_style),
                    Text(sev_label,       style=sev_style),
                    Text(cell_str,        style="dim"),
                    Text(val_str,         style=sev_style),
                    Text(fault.description, style="dim"),
                )

        # Count breakdown
        n_crit = sum(1 for f in active_faults if f.severity == 2)
        n_warn = sum(1 for f in active_faults if f.severity == 1)
        n_pred = sum(1 for f in active_faults if f.severity == 0)
        parts = []
        if n_crit:
            parts.append(f"[bold red]{n_crit} CRIT[/bold red]")
        if n_warn:
            parts.append(f"[yellow]{n_warn} WARN[/yellow]")
        if n_pred:
            parts.append(f"[magenta]{n_pred} PRED[/magenta]")
        count_str = "  ".join(parts) if parts else "[green]0 active[/green]"

        return Panel(
            tbl,
            title=f"[bold]FAULTS & PREDICTIONS  {count_str}[/bold]",
            box=box.ROUNDED,
            style="on grey11",
        )

    def _render_imd_panel(self, state: PackState) -> Panel:
        """IMD / AIR shutdown circuit status."""
        imd_style = "bold red on dark_red" if state.imd_triggered else "bold green"
        air_style = "bold red on dark_red" if not state.airs_closed else "bold green"
        imd_text = " !! TRIGGERED !! " if state.imd_triggered else "  OK  "
        air_text = " !! OPEN / TRIPPED !! " if not state.airs_closed else "  CLOSED  "

        g = Table.grid(expand=True, padding=(0, 3))
        g.add_column(justify="left")
        g.add_column(justify="left")
        g.add_row("[bold]IMD Status[/bold]",
                  Text(imd_text,  style=imd_style))
        g.add_row("[bold]AIR Status[/bold]",
                  Text(air_text,  style=air_style))
        g.add_row(
            "[dim]Iso R+ (MΩ)[/dim]",
            Text(f"{state.iso_resistance_pos:.4f}",
                 style="bold red" if state.iso_resistance_pos < 0.20 else "green"),
        )
        g.add_row(
            "[dim]Iso R− (MΩ)[/dim]",
            Text(f"{state.iso_resistance_neg:.4f}",
                 style="bold red" if state.iso_resistance_neg < 0.20 else "green"),
        )
        g.add_row("[dim]AIRs Closed[/dim]", Text(str(state.airs_closed)))

        border_style = "on dark_red" if state.imd_triggered or not state.airs_closed else "on grey11"
        return Panel(g, title="[bold]IMD / SHUTDOWN CIRCUIT[/bold]",
                     box=box.HEAVY, style=border_style)

    def _render_footer(self) -> Panel:
        keys = (
            "[bold]Q[/bold] quit   "
            "[bold]1[/bold] idle  "
            "[bold]2[/bold] cruise  "
            "[bold]3[/bold] accel  "
            "[bold]4[/bold] regen  "
            "[bold]5[/bold] charge  "
            "[bold]F[/bold] inject fault  "
            "[bold]R[/bold] reset faults  "
            "[bold]S[/bold] cell snapshot"
        )
        return Panel(
            Text(keys, justify="center"),
            style="on grey15",
            box=box.SIMPLE,
        )
