"""Integrate manager-approved settings with a dynamic virtual batch run.

This software-only simulator does not control hardware. Settings influence
illustrative process-stage timing, temperature readings, and energy estimates.
"""
from __future__ import annotations

from datetime import datetime, timezone
import pandas as pd

from src.optimization import apply_scenario, estimate_energy
from src.production_line import simulate_production

def build_run_record(
    batch_id: str,
    product_type: str,
    baseline_row: pd.DataFrame,
    approved_settings: dict,
    quality_model,
    run_number: int = 1,
    seed: int = 101,
) -> dict:
    """Run a dynamic simulated batch and return its quality/line record."""
    approved_row = apply_scenario(baseline_row, approved_settings)
    quality_estimate = float(quality_model.predict(approved_row)[0])
    quality_estimate = max(0.0, min(100.0, quality_estimate))
    energy_estimate = estimate_energy(approved_row)

    stage_events, batch_summary, metrics = simulate_production(
        products=[product_type],
        batches_per_product=1,
        seed=seed,
        machine_capacity=1,
        process_settings=approved_settings,
    )
    sim_summary = batch_summary.iloc[0].to_dict() if not batch_summary.empty else {}
    stage_rows = stage_events.to_dict(orient="records") if not stage_events.empty else []

    return {
        "run_id": f"RUN-{batch_id}-{run_number:03d}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_id,
        "product_type": product_type,
        "run_number": int(run_number),
        "settings": {k: float(approved_row.iloc[0][k]) for k in approved_settings},
        "estimated_quality_score": round(quality_estimate, 3),
        "estimated_energy_kwh": round(energy_estimate, 4),
        "simulated_line_lead_time_min": sim_summary.get("lead_time_min"),
        "simulated_line_energy_kwh": sim_summary.get("energy_kwh"),
        "simulation_batches_completed": metrics["kpis"].get("batches_completed", 0),
        "dynamic_stage_count": len(stage_rows),
        "dynamic_stage_events": stage_rows,
        "dynamic_simulation_settings_applied": metrics["kpis"].get("dynamic_settings_applied", False),
        "run_status": "SIMULATED",
        "note": "Process settings affect illustrative stage timing/temperature/energy proxies; coefficients are synthetic and not calibrated or validated for real production.",
    }
