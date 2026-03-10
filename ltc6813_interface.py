"""
ltc6813_interface.py -- LTC6813-1 Battery Stack Monitor SPI Interface
Rutgers Formula Racing BMS Fault Detector

Simulates the full SPI protocol layer for the LTC6813-1 isoSPI battery
monitor IC, matching the exact register map and command set from the
Analog Devices datasheet (Rev A).

Pipeline:  PackSimulator (physics) -> Ltc6813Interface (SPI + ADC) -> FaultDetector

Features:
  - Real LTC6813 command bytes: ADCV, RDCVA/B/C/D, ADAX, RDAUXA, WRCFGA
  - PEC15 CRC on every TX/RX frame using polynomial 0x4599
  - ADC quantization at 100 uV LSB (16-bit register model)
  - NTC temperature via Steinhart-Hart (A/B/C/D coefficients)
  - Configurable SPI fault injection: PEC mismatches, retries, COMM_FAULT
  - Discharge enable via WRCFGA DCC bits for cell balancing
  - Per-IC daisy-chain model: 6 ICs for 96 cells (18 cells/IC, but
    LTC6813 measures 18 cells; we use 16 per IC + 2 unused for 96 total)

Register layout per LTC6813:
  Cell voltages: RDCVA (cells 1-3), RDCVB (4-6), RDCVC (7-9), RDCVD (10-12)
                 RDCVE (13-15), RDCVF (16-18)
  For 96s pack with 6 ICs: IC0 cells 1-16, IC1 cells 17-32, ... IC5 cells 81-96
  GPIO/Aux:      RDAUXA (GPIO1-3), RDAUXB (GPIO4-5 + REF)

Simplification: We use 6 ICs x 16 cells = 96 series groups.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from simulator import PackState, CellState, SERIES_COUNT, PARALLEL_COUNT


# ---------------------------------------------------------------------------
# LTC6813 Command Codes (CMD[15:0] as 16-bit big-endian)
# ---------------------------------------------------------------------------

# ADC conversion commands
CMD_ADCV = 0x0360   # Start cell voltage ADC conversion (all cells, 7kHz mode)
CMD_ADAX = 0x0560   # Start GPIO/Aux ADC conversion

# Read cell voltage register groups (6 bytes per group = 3 cell voltages)
CMD_RDCVA = 0x0004   # Cell voltages 1-3
CMD_RDCVB = 0x0006   # Cell voltages 4-6
CMD_RDCVC = 0x0008   # Cell voltages 7-9
CMD_RDCVD = 0x000A   # Cell voltages 10-12
CMD_RDCVE = 0x000C   # Cell voltages 13-15
CMD_RDCVF = 0x000E   # Cell voltages 16-18

# Read auxiliary register groups
CMD_RDAUXA = 0x000C   # GPIO 1-3 (NTC temperatures)
CMD_RDAUXB = 0x000E   # GPIO 4-5 + VREF2

# Write configuration register group A
CMD_WRCFGA = 0x0001   # Configuration register A (includes DCC discharge bits)

# Cell voltage register group command list (ordered)
RDCV_COMMANDS = [CMD_RDCVA, CMD_RDCVB,
                 CMD_RDCVC, CMD_RDCVD, CMD_RDCVE, CMD_RDCVF]

# ---------------------------------------------------------------------------
# PEC15 CRC (polynomial 0x4599, initial remainder 0x0010)
# ---------------------------------------------------------------------------

# Precomputed PEC15 lookup table for polynomial x^15 + x^14 + x^10 + x^8 +
# x^7 + x^4 + x^3 + x^0 = 0x4599
_PEC15_TABLE: List[int] = []


def _init_pec15_table() -> None:
    """Build the 256-entry CRC-15 lookup table."""
    poly = 0x4599
    for i in range(256):
        remainder = i << 7
        for _ in range(8):
            if remainder & 0x4000:
                remainder = (remainder << 1) ^ poly
            else:
                remainder = remainder << 1
        _PEC15_TABLE.append(remainder & 0x7FFF)


_init_pec15_table()


def pec15(data: bytes) -> int:
    """
    Compute PEC15 checksum over a byte sequence.

    Returns the 15-bit CRC as a 16-bit integer (MSB padded).
    The LTC6813 transmits PEC as two bytes: [PEC[14:8], PEC[7:0]].
    """
    remainder = 0x0010  # initial seed per datasheet
    for byte in data:
        addr = ((remainder >> 7) ^ byte) & 0xFF
        remainder = (remainder << 8) ^ _PEC15_TABLE[addr]
    # left-shift by 1 per datasheet convention
    return (remainder << 1) & 0xFFFF


def pec15_bytes(data: bytes) -> bytes:
    """Return PEC15 as two bytes [MSB, LSB] ready to append to a frame."""
    crc = pec15(data)
    return bytes([(crc >> 8) & 0xFF, crc & 0xFF])


def verify_pec15(data: bytes, expected: bytes) -> bool:
    """Verify PEC15 of data against expected two-byte PEC."""
    return pec15_bytes(data) == expected


# ---------------------------------------------------------------------------
# ADC model constants
# ---------------------------------------------------------------------------

ADC_LSB_V = 0.0001       # 100 uV per LSB (16-bit cell voltage register)
ADC_BITS = 16
ADC_MAX_CODE = (1 << ADC_BITS) - 1

# NTC Steinhart-Hart coefficients (standard 10k NTC, Murata NCP18XH103F03RB)
NTC_A = 0.003354016
NTC_B = 0.000256985
NTC_C = 0.000002620
NTC_D = 0.000000063

# NTC voltage divider: Vref=3.0V, R_pullup=10kOhm, NTC to GND
NTC_VREF = 3.0       # V — LTC6813 VREF2
NTC_R_PULLUP = 10000.0   # ohm


# ---------------------------------------------------------------------------
# Steinhart-Hart temperature conversion
# ---------------------------------------------------------------------------

def ntc_voltage_to_temp_c(v_ntc: float) -> float:
    """
    Convert NTC voltage divider output to temperature in Celsius.

    Circuit: VREF -- R_pullup -- ADC_pin -- NTC -- GND
    v_ntc = NTC_VREF * R_ntc / (R_pullup + R_ntc)
    """
    if v_ntc <= 0.001 or v_ntc >= NTC_VREF - 0.001:
        return -999.0  # open/short circuit

    r_ntc = NTC_R_PULLUP * v_ntc / (NTC_VREF - v_ntc)

    if r_ntc <= 0:
        return -999.0

    ln_r = np.log(r_ntc / 10000.0)  # ln(R/R0) where R0=10k
    inv_t = NTC_A + NTC_B * ln_r + NTC_C * ln_r**2 + NTC_D * ln_r**3
    temp_k = 1.0 / inv_t
    return float(temp_k - 273.15)


def temp_c_to_ntc_voltage(temp_c: float) -> float:
    """
    Reverse: given a temperature in C, compute what the NTC voltage
    divider would output. Used to generate simulated ADC readings.
    """
    temp_k = temp_c + 273.15
    # Inverse Steinhart-Hart to get R_ntc
    # Simplified: use standard B-parameter approximation for reverse
    # Full reverse requires solving cubic, so we use the lookup approach
    ln_r = (1.0 / temp_k - NTC_A) / NTC_B  # first-order approx
    r_ntc = 10000.0 * np.exp(ln_r)
    r_ntc = max(r_ntc, 1.0)  # clamp
    v_ntc = NTC_VREF * r_ntc / (NTC_R_PULLUP + r_ntc)
    return float(v_ntc)


# ---------------------------------------------------------------------------
# SPI Frame structures
# ---------------------------------------------------------------------------

@dataclass
class SpiFrame:
    """One SPI transaction frame (command + response)."""
    command:       int              # 16-bit command code
    tx_data:       bytes = b""     # Data sent to IC (for WRCFGA)
    tx_pec:        bytes = b""     # PEC of tx_data
    rx_data:       bytes = b""     # Data received from IC
    rx_pec:        bytes = b""     # PEC received from IC
    pec_valid:     bool = True    # Was PEC check OK?
    ic_index:      int = 0       # Which IC in daisy chain (0-5)
    timestamp:     float = 0.0
    latency_us:    float = 0.0     # Simulated ADC conversion time


@dataclass
class SpiStats:
    """Cumulative SPI bus statistics."""
    total_frames:     int = 0
    pec_mismatches:   int = 0
    retries:          int = 0
    comm_faults:      int = 0   # unrecoverable after max retries
    total_bytes_tx:   int = 0
    total_bytes_rx:   int = 0
    last_latency_us:  float = 0.0
    avg_latency_us:   float = 0.0
    _latency_sum:     float = 0.0


# ---------------------------------------------------------------------------
# LTC6813 Interface
# ---------------------------------------------------------------------------

NUM_ICS = 6    # 6 ICs in daisy chain for 96 cells
CELLS_PER_IC = 16   # Using 16 of 18 channels per IC
GPIO_PER_IC = 5    # 5 GPIO pins per IC (NTC temperature sensors)
MAX_SPI_RETRIES = 3    # Retry on PEC mismatch before COMM_FAULT


@dataclass
class Ltc6813Config:
    """Configuration for the SPI interface simulation."""
    pec_error_rate:     float = 0.01    # 1% chance of PEC mismatch per frame
    adc_noise_uv:       float = 50.0    # ADC noise standard deviation (uV)
    adc_latency_us:     float = 2340.0  # 7kHz mode conversion time (us)
    spi_clock_hz:       int = 1000000  # 1 MHz SPI clock
    max_retries:        int = MAX_SPI_RETRIES


class Ltc6813Interface:
    """
    Simulated LTC6813-1 isoSPI interface layer.

    Sits between PackSimulator and FaultDetector:
      1. Reads ideal cell voltages/temperatures from PackState
      2. Quantizes through ADC model (100uV LSB, 16-bit)
      3. Converts NTC voltages via Steinhart-Hart
      4. Wraps in SPI frames with PEC15 checksums
      5. Injects random PEC errors at configured rate
      6. Retries on PEC mismatch, raises COMM_FAULT after max retries

    Usage:
        iface = Ltc6813Interface()
        voltages, temps, comm_ok = iface.read_all(pack_state)
    """

    def __init__(self, config: Optional[Ltc6813Config] = None, seed: int = 42):
        self.cfg = config or Ltc6813Config()
        self._rng = random.Random(seed + 99)
        self._np_rng = np.random.RandomState(seed + 99)

        self.stats = SpiStats()

        # Per-IC configuration registers (CFGA: 6 bytes each)
        # Byte 4 bits [7:0] = DCC[8:1], Byte 5 bits [3:0] = DCC[12:9]
        # DCC bits enable discharge FET for that cell
        self._cfga: List[bytearray] = [bytearray(6) for _ in range(NUM_ICS)]

        # Latest ADC results (raw 16-bit codes) per IC
        self._cell_adc: List[List[int]] = [
            [0] * CELLS_PER_IC for _ in range(NUM_ICS)]
        self._gpio_adc: List[List[int]] = [
            [0] * GPIO_PER_IC for _ in range(NUM_ICS)]

        # Track which cells have discharge enabled
        self.discharge_enabled: Dict[int, bool] = {}

        # Latest comm fault flag (cleared each read cycle)
        self.comm_fault_active: bool = False
        self._last_comm_fault_ic: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_all(self, state: PackState) -> Tuple[
        List[float],    # 96 cell voltages (V) post-ADC
        List[float],    # 96 cell temperatures (C) post-NTC
        bool,           # comm_ok (False if COMM_FAULT occurred)
    ]:
        """
        Full acquisition cycle: ADCV -> RDCVA-F -> ADAX -> RDAUXA.
        Returns quantized voltages, temperatures, and comm status.
        """
        self.comm_fault_active = False
        self._last_comm_fault_ic = None

        # Step 1: Start cell voltage ADC conversion
        self._send_command(CMD_ADCV)
        self._simulate_adc_conversion(state)

        # Step 2: Read all cell voltage register groups
        cell_voltages: List[float] = []
        for ic in range(NUM_ICS):
            ic_voltages = self._read_cell_voltages(ic)
            if ic_voltages is None:
                # COMM_FAULT on this IC
                self.comm_fault_active = True
                self._last_comm_fault_ic = ic
                # Fill with last known values or zeros
                ic_voltages = [0.0] * CELLS_PER_IC
            cell_voltages.extend(ic_voltages)

        # Step 3: Start GPIO ADC conversion (NTC temperatures)
        self._send_command(CMD_ADAX)

        # Step 4: Read auxiliary registers for temperature
        cell_temps: List[float] = []
        for ic in range(NUM_ICS):
            ic_temps = self._read_temperatures(ic, state)
            if ic_temps is None:
                self.comm_fault_active = True
                self._last_comm_fault_ic = ic
                ic_temps = [25.0] * CELLS_PER_IC  # fallback
            cell_temps.extend(ic_temps)

        return cell_voltages, cell_temps, not self.comm_fault_active

    def write_discharge_config(
        self,
        discharge_map: Dict[int, bool],
    ) -> bool:
        """
        Write CFGA register to enable/disable discharge FETs.
        discharge_map: {series_index: True/False}
        Returns True if all writes succeeded.
        """
        self.discharge_enabled = discharge_map.copy()
        all_ok = True

        for ic in range(NUM_ICS):
            # Build DCC bits for this IC's 16 cells
            dcc_bits = 0
            base_cell = ic * CELLS_PER_IC
            for local_cell in range(CELLS_PER_IC):
                global_idx = base_cell + local_cell
                if discharge_map.get(global_idx, False):
                    dcc_bits |= (1 << local_cell)

            # Pack into CFGA register format
            cfga = bytearray(6)
            cfga[0] = 0xFE          # GPIO pulldowns off, REFON=1, ADCOPT=0
            cfga[1] = 0x00          # VUV[7:0]
            cfga[2] = 0x00          # VOV[3:0] | VUV[11:8]
            cfga[3] = 0x00          # VOV[11:4]
            cfga[4] = dcc_bits & 0xFF         # DCC[8:1]
            cfga[5] = (dcc_bits >> 8) & 0xFF  # DCC[16:9]

            self._cfga[ic] = cfga

            # Send WRCFGA command with data + PEC
            ok = self._write_register(ic, CMD_WRCFGA, bytes(cfga))
            if not ok:
                all_ok = False
                self.comm_fault_active = True

        return all_ok

    def get_stats(self) -> SpiStats:
        """Return current SPI bus statistics."""
        return self.stats

    # ------------------------------------------------------------------
    # Internal: ADC simulation
    # ------------------------------------------------------------------

    def _simulate_adc_conversion(self, state: PackState) -> None:
        """
        Sample ideal cell voltages from PackState, quantize through
        the 16-bit ADC model with noise, and store in register buffers.
        """
        for ic in range(NUM_ICS):
            base_series = ic * CELLS_PER_IC
            for local_ch in range(CELLS_PER_IC):
                series_idx = base_series + local_ch
                if series_idx >= SERIES_COUNT:
                    self._cell_adc[ic][local_ch] = 0
                    continue

                # Average voltage of parallel cells in this series group
                group_cells = [
                    c for c in state.cells if c.series_index == series_idx
                ]
                if not group_cells:
                    self._cell_adc[ic][local_ch] = 0
                    continue

                # In a real parallel group the LTC6813 measures at the
                # shared node — use max() so single-cell overvoltage
                # (or min for undervoltage) is not averaged away.
                v_ideal = float(max(c.voltage for c in group_cells))

                # Add ADC noise
                noise_v = self._np_rng.normal(0, self.cfg.adc_noise_uv * 1e-6)
                v_measured = v_ideal + noise_v

                # Quantize to 100uV LSB
                adc_code = int(round(v_measured / ADC_LSB_V))
                adc_code = max(0, min(ADC_MAX_CODE, adc_code))
                self._cell_adc[ic][local_ch] = adc_code

            # GPIO channels: simulate NTC voltage from cell temperatures
            for gpio in range(GPIO_PER_IC):
                # Map GPIO to series groups (distribute evenly)
                # Each GPIO covers ~3 cells on this IC
                covered_start = base_series + gpio * \
                    (CELLS_PER_IC // GPIO_PER_IC)
                covered_end = covered_start + (CELLS_PER_IC // GPIO_PER_IC)
                covered_cells = [
                    c for c in state.cells
                    if covered_start <= c.series_index < covered_end
                ]
                if covered_cells:
                    t_max = max(c.temperature for c in covered_cells)
                else:
                    t_max = 25.0

                # Convert temperature to NTC voltage divider output
                v_ntc = temp_c_to_ntc_voltage(t_max)
                noise_v = self._np_rng.normal(0, self.cfg.adc_noise_uv * 1e-6)
                v_ntc += noise_v

                adc_code = int(round(v_ntc / ADC_LSB_V))
                adc_code = max(0, min(ADC_MAX_CODE, adc_code))
                self._gpio_adc[ic][gpio] = adc_code

    # ------------------------------------------------------------------
    # Internal: SPI frame operations
    # ------------------------------------------------------------------

    def _send_command(self, cmd: int) -> SpiFrame:
        """Send a broadcast command (no per-IC addressing for daisy chain)."""
        cmd_bytes = bytes([(cmd >> 8) & 0xFF, cmd & 0xFF])
        pec = pec15_bytes(cmd_bytes)

        frame = SpiFrame(
            command=cmd,
            tx_data=cmd_bytes + pec,
            tx_pec=pec,
            pec_valid=True,
            timestamp=time.time(),
            latency_us=self.cfg.adc_latency_us if cmd in (
                CMD_ADCV, CMD_ADAX) else 10.0,
        )

        self.stats.total_frames += 1
        self.stats.total_bytes_tx += len(frame.tx_data)
        self.stats.last_latency_us = frame.latency_us
        self.stats._latency_sum += frame.latency_us
        self.stats.avg_latency_us = self.stats._latency_sum / self.stats.total_frames

        return frame

    def _read_register_group(
        self, ic: int, cmd: int, num_bytes: int = 6,
    ) -> Optional[bytes]:
        """
        Read a register group from one IC with PEC15 verification.
        Retries up to max_retries on PEC mismatch.
        Returns None on COMM_FAULT (all retries exhausted).
        """
        for attempt in range(self.cfg.max_retries + 1):
            # Build the register data from our ADC buffers
            raw_data = self._get_register_data(ic, cmd)

            # Compute correct PEC
            correct_pec = pec15_bytes(raw_data)

            # Simulate SPI fault injection: random PEC corruption
            if self._rng.random() < self.cfg.pec_error_rate:
                # Corrupt one byte of the received PEC
                corrupted = bytearray(correct_pec)
                corrupted[self._rng.randint(
                    0, 1)] ^= self._rng.randint(1, 0xFF)
                rx_pec = bytes(corrupted)
                pec_ok = False
            else:
                rx_pec = correct_pec
                pec_ok = True

            frame = SpiFrame(
                command=cmd,
                rx_data=raw_data,
                rx_pec=rx_pec,
                pec_valid=pec_ok,
                ic_index=ic,
                timestamp=time.time(),
                latency_us=10.0 + self._rng.gauss(0, 2),
            )

            self.stats.total_frames += 1
            self.stats.total_bytes_rx += len(raw_data) + 2  # data + PEC
            self.stats.last_latency_us = frame.latency_us
            self.stats._latency_sum += frame.latency_us
            self.stats.avg_latency_us = self.stats._latency_sum / self.stats.total_frames

            if pec_ok:
                return raw_data
            else:
                self.stats.pec_mismatches += 1
                if attempt < self.cfg.max_retries:
                    self.stats.retries += 1

        # All retries exhausted -> COMM_FAULT
        self.stats.comm_faults += 1
        return None

    def _write_register(self, ic: int, cmd: int, data: bytes) -> bool:
        """
        Write a register group to one IC with PEC15.
        Returns True if write succeeded (PEC verified on echo).
        """
        pec = pec15_bytes(data)

        # Simulate potential PEC error on the write path
        if self._rng.random() < self.cfg.pec_error_rate:
            self.stats.pec_mismatches += 1
            self.stats.total_frames += 1
            # cmd + pec + data + pec
            self.stats.total_bytes_tx += len(data) + 4
            return False

        self.stats.total_frames += 1
        self.stats.total_bytes_tx += len(data) + 4
        return True

    def _get_register_data(self, ic: int, cmd: int) -> bytes:
        """
        Build the 6-byte register response for a given IC and command.
        Cell voltage registers: 3 cells x 2 bytes = 6 bytes per group.
        """
        data = bytearray(6)

        if cmd in RDCV_COMMANDS:
            # Determine which 3 cells this register group covers
            group_idx = RDCV_COMMANDS.index(cmd)
            start_ch = group_idx * 3

            for i in range(3):
                ch = start_ch + i
                if ch < CELLS_PER_IC:
                    code = self._cell_adc[ic][ch]
                else:
                    code = 0
                # Little-endian 16-bit
                data[i * 2] = code & 0xFF
                data[i * 2 + 1] = (code >> 8) & 0xFF

        elif cmd == CMD_RDAUXA:
            # GPIO 1-3: 3 channels x 2 bytes
            for i in range(3):
                if i < GPIO_PER_IC:
                    code = self._gpio_adc[ic][i]
                else:
                    code = 0
                data[i * 2] = code & 0xFF
                data[i * 2 + 1] = (code >> 8) & 0xFF

        elif cmd == CMD_RDAUXB:
            # GPIO 4-5 + VREF2
            for i in range(2):
                idx = 3 + i
                if idx < GPIO_PER_IC:
                    code = self._gpio_adc[ic][idx]
                else:
                    code = 0
                data[i * 2] = code & 0xFF
                data[i * 2 + 1] = (code >> 8) & 0xFF
            # VREF2 (3.0V nominal)
            vref_code = int(round(NTC_VREF / ADC_LSB_V))
            data[4] = vref_code & 0xFF
            data[5] = (vref_code >> 8) & 0xFF

        return bytes(data)

    # ------------------------------------------------------------------
    # Internal: decode register data to engineering units
    # ------------------------------------------------------------------

    def _read_cell_voltages(self, ic: int) -> Optional[List[float]]:
        """
        Read all 16 cell voltages for one IC by reading register groups
        RDCVA through RDCVF, verifying PEC on each.
        Returns list of 16 voltages in Volts, or None on COMM_FAULT.
        """
        voltages: List[float] = []

        for cmd in RDCV_COMMANDS:
            raw = self._read_register_group(ic, cmd)
            if raw is None:
                return None  # COMM_FAULT

            # Decode 3 cells from 6 bytes (little-endian 16-bit)
            for i in range(3):
                code = raw[i * 2] | (raw[i * 2 + 1] << 8)
                v = code * ADC_LSB_V
                voltages.append(v)

        # We only use the first 16 channels (LTC6813 has 18)
        return voltages[:CELLS_PER_IC]

    def _read_temperatures(
        self, ic: int, state: PackState,
    ) -> Optional[List[float]]:
        """
        Read NTC temperatures for all cells covered by this IC.
        Reads RDAUXA (GPIO1-3) and RDAUXB (GPIO4-5), converts via
        Steinhart-Hart, then maps to per-cell temperatures.
        """
        raw_a = self._read_register_group(ic, CMD_RDAUXA)
        if raw_a is None:
            return None

        raw_b = self._read_register_group(ic, CMD_RDAUXB)
        if raw_b is None:
            return None

        # Decode GPIO voltages
        gpio_temps: List[float] = []
        for i in range(3):
            code = raw_a[i * 2] | (raw_a[i * 2 + 1] << 8)
            v_ntc = code * ADC_LSB_V
            gpio_temps.append(ntc_voltage_to_temp_c(v_ntc))

        for i in range(2):
            code = raw_b[i * 2] | (raw_b[i * 2 + 1] << 8)
            v_ntc = code * ADC_LSB_V
            gpio_temps.append(ntc_voltage_to_temp_c(v_ntc))

        # Map 5 GPIO temps to 16 cells (each GPIO covers ~3 cells)
        cell_temps: List[float] = []
        cells_per_gpio = CELLS_PER_IC // GPIO_PER_IC  # 3 (with 1 extra)
        for local_ch in range(CELLS_PER_IC):
            gpio_idx = min(local_ch // cells_per_gpio, GPIO_PER_IC - 1)
            cell_temps.append(gpio_temps[gpio_idx])

        return cell_temps

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def build_command_frame(self, cmd: int) -> bytes:
        """Build a complete SPI command frame: CMD[15:8] CMD[7:0] PEC[15:8] PEC[7:0]."""
        cmd_bytes = bytes([(cmd >> 8) & 0xFF, cmd & 0xFF])
        return cmd_bytes + pec15_bytes(cmd_bytes)

    def reset_stats(self) -> None:
        """Reset all SPI statistics counters."""
        self.stats = SpiStats()
