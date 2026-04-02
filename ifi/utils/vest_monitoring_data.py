#!/usr/bin/env python3
"""
VEST monitoring data helpers
============================

Translate the post-processing logic used by the MATLAB shot monitoring scripts
into local, non-destructive Python helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, decimate, filtfilt, find_peaks, spectrogram

from .io_h5 import h5_safe_name
from .io_process_common import create_named_dataset
from .vest_conversion import append_vest_frames_to_results_h5, compute_vest_converted_fields
from .vest_fieldcode import format_vest_field_label, load_vest_field_maps
from .vest_utils import infer_sample_rate_from_index

MUZERO = 4 * np.pi * 1e-7
TURN_TF = 24.0
TF_GAIN = -3.0e4
TF_AXIS_R = 0.4
IND_TF = 9.3e-4
RES_TF = 0.0279
CAP_TF = 120.0
ROGO_L = 8.12e-3
ROGO_GAIN = -1.0 / ROGO_L
MONITORING_ROOT = "vest_monitoring"


def _normalize_unit_text(unit: str) -> str:
    text = str(unit).strip()
    return text[1:-1] if text.startswith("[") and text.endswith("]") else text


def _tokenize_label(text: str) -> str:
    return (
        str(text)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )


@dataclass
class VestSeries:
    field_id: int
    field_name: str
    field_unit: str
    rate_key: str
    sample_rate: float
    time: np.ndarray
    data: np.ndarray

    @property
    def label(self) -> str:
        return format_vest_field_label(self.field_name, _normalize_unit_text(self.field_unit))


def _butter_filter(
    data: np.ndarray,
    sample_rate: float,
    cutoff_hz: float,
    btype: str,
    order: int = 4,
) -> np.ndarray:
    if sample_rate <= 0 or len(data) < 16:
        return data
    wn = cutoff_hz / (0.5 * sample_rate)
    wn = min(max(wn, 1e-6), 0.999999)
    b, a = butter(order, wn, btype=btype)
    try:
        return filtfilt(b, a, data)
    except ValueError:
        return data


def _moving_median(data: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(np.asarray(data, dtype=float))
    return series.rolling(window=window, center=True, min_periods=1).median().to_numpy()


def _find_plasma_window(time: np.ndarray, data: np.ndarray) -> tuple[float, float]:
    if len(time) == 0:
        return 0.0, 0.0
    mask = (time >= 0.28) & (time <= 0.40)
    if not np.any(mask):
        return float(time[0]), float(time[-1])
    t_sel = time[mask]
    d_sel = _moving_median(np.asarray(data, dtype=float), 100)[mask]
    if len(d_sel) == 0:
        return float(time[0]), float(time[-1])
    amplitude = np.nanmax(d_sel) - np.nanmin(d_sel)
    if not np.isfinite(amplitude) or amplitude <= 0:
        return float(t_sel[0]), float(t_sel[-1])
    threshold = np.nanmin(d_sel) + 0.02 * amplitude
    active = d_sel <= threshold if abs(np.nanmin(d_sel)) > abs(np.nanmax(d_sel)) else d_sel >= threshold
    if not np.any(active):
        peak_idx = int(np.nanargmax(np.abs(d_sel)))
        t_peak = float(t_sel[peak_idx])
        return t_peak, t_peak
    idx = np.flatnonzero(active)
    return float(t_sel[idx[0]]), float(t_sel[idx[-1]])


def _pressure_from_raw(shot_num: int, raw: VestSeries, gas: str) -> np.ndarray:
    raw_values = np.asarray(raw.data, dtype=float)
    if shot_num < 46993:
        pressure = np.power(10.0, 1.667 * raw_values - 11.46)
        window = 51
    elif 46993 < shot_num < 47176:
        pressure = np.power(10.0, 2.0 * raw_values - 10.625)
        window = 101
    elif shot_num < 47252 and raw.field_id == 219:
        pressure = np.power(10.0, 2.0 * raw_values - 10.625)
        window = 101
    else:
        pressure = np.power(10.0, 2.0 * raw_values - 10.625)
        window = 101
    factor = 2.4 if gas.upper() == "H2" else 5.9
    return _moving_median(factor * pressure, window)


def _tf_current(raw_tf: VestSeries) -> tuple[np.ndarray, np.ndarray]:
    tf = (raw_tf.data - np.mean(raw_tf.data[:25])) * TF_GAIN
    bt_r = MUZERO * tf * TURN_TF / (2.0 * np.pi)
    bt = bt_r / TF_AXIS_R
    return tf, bt


def _pf_gains(shot_num: int, xcoil: list[int]) -> list[float]:
    gains: list[float] = []
    for coil in xcoil:
        if coil == 1:
            if shot_num < 20259:
                gains.append(1e4)
            elif shot_num < 38361:
                gains.append(-5e4)
            elif shot_num < 38401:
                gains.append(5e4)
            elif shot_num < 45935:
                gains.append(-5e4)
            else:
                gains.append(-1e4)
        elif coil == 2:
            gains.append(1e3)
        elif coil == 5:
            gains.append(1e4 if shot_num < 38110 else -1e4)
        elif coil == 6:
            gains.append(1e3 if shot_num < 38110 else -1e3)
        elif coil in {9, 10}:
            gains.append(-1e3 if shot_num < 19287 else -0.5e3)
        else:
            gains.append(1.0)
    return gains


def _limiter_field_ids(shot_num: int) -> list[int]:
    if shot_num <= 45264:
        return [216, 218, 217]
    if shot_num <= 47176:
        return [217, 218, 216]
    if shot_num <= 47251:
        return [217, 12, 216]
    return [217, 218, 216]


def load_vest_series_map(
    vest_db,
    shot_num: int,
    fields: list[int],
) -> dict[int, VestSeries]:
    grouped = vest_db.load_shot(shot=shot_num, fields=fields)
    by_id, _ = load_vest_field_maps()
    loaded: dict[int, VestSeries] = {}
    for rate_key, df in grouped.items():
        sample_rate = infer_sample_rate_from_index(df.index) or 0.0
        for field_id in fields:
            meta = by_id.get(field_id, {})
            field_name = str(meta.get("field_name", field_id))
            field_unit = str(meta.get("field_unit", ""))
            norm_unit = _normalize_unit_text(field_unit)
            candidates = [
                format_vest_field_label(field_name, field_unit),
                format_vest_field_label(field_name, norm_unit),
                str(field_id),
                field_name,
            ]
            matched_col = None
            for cand in candidates:
                if cand in df.columns:
                    matched_col = cand
                    break
            if matched_col is None:
                field_token = _tokenize_label(field_name)
                for col in df.columns:
                    col_token = _tokenize_label(col)
                    if col_token.startswith(field_token):
                        matched_col = str(col)
                        break
            if matched_col is not None:
                loaded[field_id] = VestSeries(
                    field_id=field_id,
                    field_name=field_name,
                    field_unit=field_unit,
                    rate_key=rate_key,
                    sample_rate=float(sample_rate),
                    time=df.index.to_numpy(dtype=float),
                    data=df[matched_col].to_numpy(dtype=float),
                )
    return loaded


def _converted_input_map(series_map: dict[int, VestSeries]) -> dict[int, dict[str, Any]]:
    return {
        field_id: {
            "time": series.time,
            "data": series.data,
            "rate_key": series.rate_key,
        }
        for field_id, series in series_map.items()
    }


def compute_monitoring_frames(
    shot_num: int,
    series_map: dict[int, VestSeries],
    xcoil: list[int],
    gas: str,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}

    if 1 in series_map:
        tf, bt = _tf_current(series_map[1])
        frames["tf_current"] = pd.DataFrame(
            {"I_TF": tf / 1000.0, "B_T": bt},
            index=series_map[1].time * 1000.0,
        )

    pf_fields = [59, 4, 5, 62, 63, 6, 64, 65]
    pf_present = [field for field in pf_fields if field in series_map]
    if pf_present:
        time_axis = series_map[pf_present[0]].time * 1000.0
        pf_frame = pd.DataFrame(index=time_axis)
        gains = _pf_gains(shot_num, xcoil)
        field_map = {
            1: 59,
            2: 4,
            5: 5,
            6: 62,
            7: 63,
            8: 6,
            9: 64 if shot_num < 19287 else 65,
            10: 65,
        }
        for coil, gain in zip(xcoil, gains, strict=False):
            field_id = field_map.get(coil)
            if field_id not in series_map:
                continue
            data = series_map[field_id].data
            proc = (data - np.mean(data[:25])) * gain
            proc = _butter_filter(proc, series_map[field_id].sample_rate, 2500.0, "low")
            pf_frame[f"PF{coil}"] = proc / 1000.0
        if not pf_frame.empty:
            frames["pf_current"] = pf_frame

    converted_frames = compute_vest_converted_fields(shot_num, _converted_input_map(series_map))
    converted_series_map = {}
    for _, df in converted_frames.items():
        for col in df.columns:
            converted_series_map[col] = df[col]

    ip_raw = None
    if 102 in series_map:
        ip_raw = series_map[102]
    elif "Ip [kA]" in converted_series_map:
        ip_series = converted_series_map["Ip [kA]"]
        ip_raw = VestSeries(102, "Ip", "kA", "25k", 25e3, ip_series.index.to_numpy(float), ip_series.to_numpy(float))
    if ip_raw is None and 109 in series_map:
        conv = compute_vest_converted_fields(shot_num, {109: _converted_input_map(series_map)[109]})
        for _, df in conv.items():
            if "Ip [kA]" in df.columns:
                ip_series = df["Ip [kA]"]
                ip_raw = VestSeries(102, "Ip", "kA", "25k", 25e3, ip_series.index.to_numpy(float), ip_series.to_numpy(float))
                break

    if ip_raw is not None and 25 in series_map:
        time = ip_raw.time
        if shot_num >= 43635:
            x_time = slice(6500, min(9000, len(time)))
        elif (41446 <= shot_num <= 41451) or shot_num >= 41660:
            x_time = slice(7250, min(8750, len(time)))
        else:
            x_time = slice(6000, min(8500, len(time)))
        base_idx = slice(max(0, x_time.start - 500), x_time.start + 1)
        ip_shot = ip_raw.data - np.polyval(np.polyfit(time[base_idx], ip_raw.data[base_idx], 1), time)
        flux_loop = series_map[25]
        ind_mutual = 2.8e-4 if shot_num < 17455 else 5.0e-4
        if shot_num >= 46403:
            fl_time = flux_loop.time + 0.26
            fl_data = decimate(flux_loop.data, 10, zero_phase=True)
            fl_time = fl_time[::10][: len(fl_data)]
            ip_ref = np.interp(time, fl_time, _moving_median(fl_data * 11.0 / ind_mutual, 10), left=0.0, right=0.0)
            ip = ip_shot.copy()
            idx = (time >= 0.26) & (time <= 0.36)
            if shot_num < 47100:
                ip[idx] = ip[idx] - ip_ref[idx]
        else:
            fl_ref = flux_loop.data * 11.0 / ind_mutual
            fl_ref = fl_ref - np.polyval(np.polyfit(flux_loop.time[base_idx], fl_ref[base_idx], 1), flux_loop.time)
            fl_ref = _moving_median(fl_ref, 10)
            ip = ip_shot - fl_ref
        if shot_num >= 20259:
            ip = -ip
        frames["plasma_current"] = pd.DataFrame({"Ip": ip / 1000.0}, index=time * 1000.0)

    pressure_field = 219 if shot_num < 47252 and 219 in series_map and shot_num >= 47176 else 12
    if pressure_field in series_map:
        pressure = _pressure_from_raw(shot_num, series_map[pressure_field], gas)
        frames["pressure"] = pd.DataFrame({"Pressure": pressure}, index=series_map[pressure_field].time * 1000.0)

    if 25 in series_map:
        frames["flux_loop"] = pd.DataFrame(
            {"FluxLoop": series_map[25].data / (-0.0909)},
            index=260.0 + series_map[25].time * 1000.0,
        )

    filterscope2_ids = [138, 139, 140, 141, 142, 143, 144]
    if all(field in series_map for field in filterscope2_ids):
        time_shift = 0.24 if shot_num < 41660 else 0.26
        frame = pd.DataFrame(index=(series_map[138].time + time_shift) * 1000.0)
        for idx, field_id in enumerate(filterscope2_ids, start=1):
            frame[f"CH{idx:02d}"] = -series_map[field_id].data
        frames["filterscope_2"] = frame

    filterscope1_ids = [101, 214]
    if all(field in series_map for field in filterscope1_ids):
        frame = pd.DataFrame(index=series_map[101].time * 1000.0)
        frame["H_alpha"] = -series_map[101].data
        frame["OI_777"] = -series_map[214].data
        frames["filterscope_1"] = frame

    mirnov_ids = [192, 171]
    if all(field in series_map for field in mirnov_ids):
        frame = pd.DataFrame(index=series_map[192].time * 1000.0)
        frame["Mirnov_Inboard"] = series_map[192].data - np.mean(series_map[192].data)
        frame["Mirnov_Outboard"] = series_map[171].data - np.mean(series_map[171].data)
        frames["mirnov"] = frame

    limiter_ids = _limiter_field_ids(shot_num)
    if all(field in series_map for field in limiter_ids):
        frame = pd.DataFrame(index=series_map[limiter_ids[0]].time * 1000.0)
        frame["UpperConer"] = series_map[limiter_ids[0]].data
        frame["MidplaneMain"] = series_map[limiter_ids[1]].data
        frame["LowerConer"] = series_map[limiter_ids[2]].data
        frames["limiter_current"] = frame

    dia_field = 246 if shot_num < 37505 else 4 if shot_num < 38452 else 257
    if dia_field in series_map and 101 in series_map:
        temp_time = series_map[dia_field].time
        temp_raw = series_map[dia_field].data
        temp_int = np.cumsum(np.gradient(temp_time) * temp_raw) * ROGO_GAIN * 25.0
        onset, offset = _find_plasma_window(series_map[101].time, series_map[101].data)
        if onset < offset:
            idx_start = int(np.argmin(np.abs(temp_time - onset)))
            idx_end = int(np.argmin(np.abs(temp_time - offset)))
            ref_base = np.interp(
                temp_time,
                np.concatenate([temp_time[: idx_start + 1], temp_time[idx_end:]]),
                np.concatenate([temp_int[: idx_start + 1], temp_int[idx_end:]]),
            )
            delta_i = temp_int - ref_base
            dia = (
                IND_TF / TURN_TF * delta_i
                + RES_TF / TURN_TF * np.cumsum(np.gradient(temp_time) * delta_i)
                + 1.0 / CAP_TF / TURN_TF * np.cumsum(np.gradient(temp_time) * np.cumsum(np.gradient(temp_time) * delta_i))
            )
            coeff = np.polyfit(
                [temp_time[idx_start], temp_time[idx_end]],
                [dia[idx_start], dia[idx_end]],
                1,
            )
            baseline = np.polyval(coeff, temp_time)
            baseline[:idx_start] = 0.0
            dia_final = dia - baseline
            dia_final[idx_end:] = 0.0
            frames["diamagnetic_flux"] = pd.DataFrame({"DiaFlux": dia_final * 1000.0}, index=temp_time * 1000.0)

    return frames


def compute_mirnov_spectrograms(
    shot_num: int,
    monitoring_frames: dict[str, pd.DataFrame],
) -> dict[str, dict[str, np.ndarray]]:
    frame = monitoring_frames.get("mirnov")
    if frame is None or frame.empty or len(frame.index) < 16:
        return {}
    sample_rate = infer_sample_rate_from_index(pd.Index(frame.index.to_numpy(dtype=float) / 1000.0))
    if not sample_rate or sample_rate <= 0:
        return {}
    specs: dict[str, dict[str, np.ndarray]] = {}
    for col in frame.columns:
        freq, time_axis, power = spectrogram(
            frame[col].to_numpy(dtype=float),
            fs=sample_rate,
            nperseg=min(500, len(frame[col])),
            noverlap=min(400, max(0, len(frame[col]) - 1)),
            scaling="density",
            mode="magnitude",
        )
        specs[col] = {
            "shot_num": np.array([shot_num], dtype=int),
            "time_ms": time_axis * 1000.0 + frame.index.min(),
            "freq_khz": freq / 1000.0,
            "power": power,
        }
    return specs


def append_monitoring_frames_to_h5(
    h5_path: str | Path,
    monitoring_frames: dict[str, pd.DataFrame],
    overwrite: bool = False,
) -> int:
    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    written = 0
    with h5py.File(path, "a") as hf:
        root = hf.require_group(MONITORING_ROOT)
        for name, df in monitoring_frames.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            if name in root:
                if not overwrite:
                    continue
                del root[name]
            grp = root.create_group(name)
            grp.attrs["index_unit"] = "ms"
            grp.attrs["t_start"] = float(df.index.min())
            grp.attrs["t_end"] = float(df.index.max())
            grp.create_dataset("time_ms", data=df.index.to_numpy(dtype=float))
            used_names: set[str] = set(grp.keys())
            for col in df.columns:
                create_named_dataset(
                    grp,
                    str(col),
                    df[col].to_numpy(dtype=float),
                    used_names=used_names,
                    original_name=str(col),
                )
            written += 1
    return written


def append_mirnov_spectrograms_to_h5(
    h5_path: str | Path,
    spectrograms: dict[str, dict[str, np.ndarray]],
    overwrite: bool = False,
) -> int:
    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    written = 0
    with h5py.File(path, "a") as hf:
        root = hf.require_group(MONITORING_ROOT).require_group("mirnov_spectrogram")
        for name, payload in spectrograms.items():
            safe_name = h5_safe_name(name.replace(" ", "_"))
            if safe_name in root:
                if not overwrite:
                    continue
                del root[safe_name]
            grp = root.create_group(safe_name)
            for key, value in payload.items():
                grp.create_dataset(key, data=value)
            written += 1
    return written


def run_vest_monitoring_data_pipeline(
    vest_db,
    shot_num: int,
    h5_path: str | Path,
    xcoil: list[int],
    gas: str = "H2",
    overwrite_local: bool = False,
) -> dict[str, Any]:
    required_fields = sorted(
        {
            1, 4, 5, 6, 12, 25, 59, 62, 63, 64, 65, 101, 109, 138, 139, 140, 141,
            142, 143, 144, 171, 192, 214, 216, 217, 218, 219, 246, 257,
            27, 28, 96, 97, 98, 159, 160, 161, 162, 228, 229, 230, 231,
            259, 260,
        }
    )
    series_map = load_vest_series_map(vest_db, shot_num, required_fields)
    converted_frames = compute_vest_converted_fields(shot_num, _converted_input_map(series_map))
    monitoring_frames = compute_monitoring_frames(shot_num, series_map, xcoil=xcoil, gas=gas)
    mirnov_specs = compute_mirnov_spectrograms(shot_num, monitoring_frames)
    converted_written = append_vest_frames_to_results_h5(
        h5_path=h5_path,
        shot_num=shot_num,
        vest_by_rate=converted_frames,
        overwrite=overwrite_local,
    )
    monitoring_written = append_monitoring_frames_to_h5(
        h5_path=h5_path,
        monitoring_frames=monitoring_frames,
        overwrite=overwrite_local,
    )
    specs_written = append_mirnov_spectrograms_to_h5(
        h5_path=h5_path,
        spectrograms=mirnov_specs,
        overwrite=overwrite_local,
    )
    return {
        "series_map": series_map,
        "converted_frames": converted_frames,
        "monitoring_frames": monitoring_frames,
        "mirnov_spectrograms": mirnov_specs,
        "written": {
            "converted": converted_written,
            "monitoring": monitoring_written,
            "mirnov_spectrogram": specs_written,
        },
    }


__all__ = [
    "MONITORING_ROOT",
    "VestSeries",
    "append_mirnov_spectrograms_to_h5",
    "append_monitoring_frames_to_h5",
    "compute_mirnov_spectrograms",
    "compute_monitoring_frames",
    "load_vest_series_map",
    "run_vest_monitoring_data_pipeline",
]
