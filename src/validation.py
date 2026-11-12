"""Repeatability and one-factor-at-a-time sensitivity analysis.

All results quantify behavior of the project's synthetic model/simulator.
They do not validate real manufacturing processes.
"""
from __future__ import annotations

import pandas as pd

from src.integration import build_run_record

SETTING_KEYS = [
    "process_temperature_c",
    "process_time_sec",
    "slice_thickness_mm",
    "pretreatment_time_min",
    "line_speed_m_per_s",
]

def run_repeatability(
    batch_id: str,
    product_type: str,
    baseline_row: pd.DataFrame,
    settings: dict,
    quality_model,
    repetitions: int = 5,
    seed_start: int = 500,
) -> pd.DataFrame:
    """Repeat the same settings with different seeds to estimate simulation variability."""
    records = []
    for i in range(max(1, int(repetitions))):
        record = build_run_record(
            batch_id=batch_id,
            product_type=product_type,
            baseline_row=baseline_row,
            approved_settings=settings,
            quality_model=quality_model,
            run_number=i + 1,
            seed=seed_start + i,
        )
        records.append({
            "replicate": i + 1,
            "seed": seed_start + i,
            "quality_estimate": record["estimated_quality_score"],
            "model_energy_estimate_kwh": record["estimated_energy_kwh"],
            "line_energy_simulated_kwh": record["simulated_line_energy_kwh"],
            "line_lead_time_min": record["simulated_line_lead_time_min"],
            "dynamic_stage_count": record["dynamic_stage_count"],
        })
    return pd.DataFrame(records)

def summarize_repeatability(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize replicate variability using mean, standard deviation, min, max and CV."""
    metrics = [
        "quality_estimate", "model_energy_estimate_kwh",
        "line_energy_simulated_kwh", "line_lead_time_min",
    ]
    rows = []
    for metric in metrics:
        values = pd.to_numeric(results[metric], errors="coerce").dropna()
        mean = float(values.mean()) if len(values) else 0.0
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append({
            "metric": metric,
            "mean": round(mean, 4),
            "std_dev": round(std, 4),
            "minimum": round(float(values.min()), 4) if len(values) else None,
            "maximum": round(float(values.max()), 4) if len(values) else None,
            "coefficient_of_variation_pct": round(std / abs(mean) * 100, 3) if mean else 0.0,
            "replicates": int(len(values)),
        })
    return pd.DataFrame(rows)

def run_one_factor_sensitivity(
    batch_id: str,
    product_type: str,
    baseline_row: pd.DataFrame,
    base_settings: dict,
    quality_model,
    variation_pct: float = 10.0,
    points: int = 5,
    seed: int = 700,
) -> pd.DataFrame:
    """Vary one parameter at a time around a baseline and record simulated responses."""
    from src.optimization import apply_scenario, estimate_energy
    from src.production_line import simulate_production

    variation = max(1.0, min(40.0, float(variation_pct))) / 100.0
    points = max(3, min(9, int(points)))
    offsets = [(-variation + (2 * variation * i / (points - 1))) for i in range(points)]
    rows = []
    for setting_key in SETTING_KEYS:
        base_value = float(base_settings[setting_key])
        for index, offset in enumerate(offsets):
            candidate = dict(base_settings)
            candidate[setting_key] = max(0.0, base_value * (1.0 + offset))
            if setting_key == "line_speed_m_per_s":
                candidate[setting_key] = max(0.1, candidate[setting_key])
            if setting_key == "process_temperature_c":
                candidate[setting_key] = max(1.0, candidate[setting_key])
            if setting_key == "process_time_sec":
                candidate[setting_key] = max(1.0, candidate[setting_key])
            candidate_row = apply_scenario(baseline_row, candidate)
            quality = float(quality_model.predict(candidate_row)[0])
            quality = max(0.0, min(100.0, quality))
            energy_estimate = float(estimate_energy(candidate_row))
            _, summary, metrics = simulate_production(
                products=[product_type], batches_per_product=1,
                seed=seed + index, machine_capacity=1,
                process_settings=candidate,
            )
            line = summary.iloc[0] if not summary.empty else {}
            rows.append({
                "setting": setting_key,
                "setting_value": round(float(candidate[setting_key]), 4),
                "relative_change_pct": round(offset * 100, 2),
                "quality_estimate": round(quality, 4),
                "model_energy_estimate_kwh": round(energy_estimate, 5),
                "line_energy_simulated_kwh": round(float(line.get("energy_kwh", 0.0)), 5),
                "line_lead_time_min": round(float(line.get("lead_time_min", 0.0)), 4),
                "simulated_batches_completed": metrics["kpis"].get("batches_completed", 0),
                "seed": seed + index,
            })
    return pd.DataFrame(rows)
