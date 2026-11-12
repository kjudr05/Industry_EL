"""What-if analysis and constrained optimization for virtual chip batches.

This is an optimization wrapper around the trained quality model. It searches
within configurable illustrative bounds and reports model estimates. Bounds and
energy estimates are not validated manufacturing instructions.
"""
from __future__ import annotations

import itertools
import numpy as np
import pandas as pd

from src.ml_models import FEATURES

CONTROLLABLE = [
    "process_temperature_c",
    "process_time_sec",
    "slice_thickness_mm",
    "pretreatment_time_min",
    "line_speed_m_per_s",
]

def make_baseline_row(source_row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([{col: source_row[col] for col in FEATURES}], columns=FEATURES)

def apply_scenario(baseline: pd.DataFrame, changes: dict) -> pd.DataFrame:
    row = baseline.copy()
    for key, value in changes.items():
        if key in row.columns:
            row.loc[row.index[0], key] = value
    return row

def scenario_grid(product: str, baseline: pd.DataFrame, temp_steps: int = 5, time_steps: int = 5):
    """Create a modest product-aware grid around baseline settings."""
    temp = float(baseline.iloc[0]["process_temperature_c"])
    duration = float(baseline.iloc[0]["process_time_sec"])
    thickness = float(baseline.iloc[0]["slice_thickness_mm"])
    pretime = float(baseline.iloc[0]["pretreatment_time_min"])
    speed = float(baseline.iloc[0]["line_speed_m_per_s"])

    if product == "Baked Potato Chips":
        temp_low, temp_high = 160.0, 220.0
        time_low, time_high = 300.0, 620.0
    elif product == "Banana Chips":
        temp_low, temp_high = 140.0, 185.0
        time_low, time_high = 65.0, 190.0
    else:
        temp_low, temp_high = 145.0, 200.0
        time_low, time_high = 75.0, 210.0

    temps = np.linspace(max(temp_low, temp - 15), min(temp_high, temp + 15), temp_steps)
    times = np.linspace(max(time_low, duration - 35), min(time_high, duration + 35), time_steps)
    thicknesses = np.linspace(max(0.7, thickness - 0.25), min(2.4, thickness + 0.25), 3)
    pre_times = np.linspace(max(0, pretime - 2), min(15, pretime + 2), 2)
    speeds = np.linspace(max(0.5, speed - 0.2), min(1.5, speed + 0.2), 2)

    for t, d, thick, pre, spd in itertools.product(temps, times, thicknesses, pre_times, speeds):
        yield {
            "process_temperature_c": float(t),
            "process_time_sec": float(d),
            "slice_thickness_mm": float(thick),
            "pretreatment_time_min": float(pre),
            "line_speed_m_per_s": float(spd),
        }

def estimate_energy(row: pd.DataFrame) -> float:
    """Illustrative energy proxy; not a calibrated physical energy model."""
    r = row.iloc[0]
    if r["product_type"] == "Baked Potato Chips":
        return float(1.5 + r["process_temperature_c"] * r["process_time_sec"] / 50000 + r["line_speed_m_per_s"] * 0.4)
    if r["product_type"] == "Banana Chips":
        return float(2.0 + r["process_temperature_c"] * r["process_time_sec"] / 20000 + r["line_speed_m_per_s"] * 0.45)
    return float(2.2 + r["process_temperature_c"] * r["process_time_sec"] / 18000 + r["line_speed_m_per_s"] * 0.5)

def run_what_if(model, baseline: pd.DataFrame, changes: dict) -> dict:
    candidate = apply_scenario(baseline, changes)
    quality = float(np.asarray(model.predict(candidate)).reshape(-1)[0])
    energy = estimate_energy(candidate)
    return {
        "scenario": candidate.iloc[0].to_dict(),
        "quality_estimate": float(np.clip(quality, 0, 100)),
        "energy_estimate_kwh": energy,
    }

def optimize_settings(model, baseline: pd.DataFrame, target_quality: float = 85.0, energy_weight: float = 0.15):
    """Grid-search feasible candidate settings, prioritizing target quality and energy.

    Ranking objective:
    - If candidates reach target, prefer lower energy and quality closer to target.
    - Otherwise, prefer higher predicted quality with a small energy penalty.
    """
    product = str(baseline.iloc[0]["product_type"])
    rows = []
    for changes in scenario_grid(product, baseline):
        candidate = apply_scenario(baseline, changes)
        quality = float(np.clip(np.asarray(model.predict(candidate)).reshape(-1)[0], 0, 100))
        energy = estimate_energy(candidate)
        reaches_target = quality >= target_quality
        if reaches_target:
            objective = 1000 - energy * energy_weight - abs(quality - target_quality) * 0.1
        else:
            objective = quality - energy_weight * energy
        rows.append({
            **changes,
            "predicted_quality": quality,
            "estimated_energy_kwh": energy,
            "target_reached": reaches_target,
            "objective_score": objective,
        })
    results = pd.DataFrame(rows).sort_values("objective_score", ascending=False).reset_index(drop=True)
    best = results.iloc[0].to_dict() if not results.empty else {}
    return results, best
