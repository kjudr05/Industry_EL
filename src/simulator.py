"""Transparent synthetic data generator for the virtual chip factory.

All equations here are prototype assumptions, not validated food-engineering laws.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRODUCTS = ["Fried Potato Chips", "Baked Potato Chips", "Banana Chips"]

def _clip(x, low, high):
    return np.clip(x, low, high)

def generate_batches(n_per_product: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic batches with product-specific process settings."""
    rng = np.random.default_rng(seed)
    rows = []
    batch_no = 1

    for product in PRODUCTS:
        for _ in range(n_per_product):
            ambient_temp = rng.uniform(20, 38)
            humidity = rng.uniform(30, 85)
            line_speed = rng.uniform(0.6, 1.4)
            slice_thickness = rng.uniform(0.8, 2.2)

            if product == "Fried Potato Chips":
                raw = rng.uniform(68, 82)  # assumed raw potato moisture (%)
                process_temp = rng.normal(175, 8)
                process_time = rng.normal(145, 18)
                oil_quality = rng.uniform(55, 100)
                pretreatment_time = rng.uniform(0, 8)
                energy_kwh = 2.2 + process_temp * process_time / 18000 + line_speed * 0.5
                # Distance from a synthetic target, not a real recipe.
                temp_penalty = abs(process_temp - 178) / 10
                time_penalty = abs(process_time - 145) / 25
                moisture_penalty = abs(raw - 74) / 5
                oil_penalty = (100 - oil_quality) / 18
                thickness_penalty = abs(slice_thickness - 1.4) / 0.35
                process_name = "Frying"
            elif product == "Baked Potato Chips":
                raw = rng.uniform(68, 82)
                process_temp = rng.normal(190, 12)
                process_time = rng.normal(480, 55)
                oil_quality = np.nan  # not a primary process input for this variant
                pretreatment_time = rng.uniform(0, 8)
                energy_kwh = 1.5 + process_temp * process_time / 50000 + line_speed * 0.4
                temp_penalty = abs(process_temp - 190) / 15
                time_penalty = abs(process_time - 480) / 70
                moisture_penalty = abs(raw - 74) / 5
                oil_penalty = 0.0
                thickness_penalty = abs(slice_thickness - 1.3) / 0.4
                process_name = "Baking"
            else:
                raw = rng.uniform(55, 72)  # assumed banana moisture (%)
                process_temp = rng.normal(165, 9)
                process_time = rng.normal(125, 20)
                oil_quality = rng.uniform(55, 100)
                pretreatment_time = rng.uniform(0, 12)
                energy_kwh = 2.0 + process_temp * process_time / 20000 + line_speed * 0.45
                temp_penalty = abs(process_temp - 165) / 11
                time_penalty = abs(process_time - 125) / 28
                moisture_penalty = abs(raw - 63.5) / 5
                oil_penalty = (100 - oil_quality) / 20
                thickness_penalty = abs(slice_thickness - 1.5) / 0.35
                process_name = "Frying"

            process_temp = float(_clip(process_temp, 120, 240))
            process_time = float(_clip(process_time, 40, 650))
            motor_vibration = float(_clip(rng.normal(2.5 + line_speed * 0.8, 0.7), 0.3, 7.0))
            machine_efficiency = float(_clip(100 - motor_vibration * 3 - abs(line_speed - 1.0) * 10, 50, 100))

            # Transparent synthetic quality equation: designed for prototyping only.
            environmental_penalty = abs(ambient_temp - 27) / 8 + abs(humidity - 55) / 25
            noise = rng.normal(0, 3.0)
            quality = 96 - (
                4.2 * temp_penalty
                + 3.8 * time_penalty
                + 3.4 * moisture_penalty
                + 2.2 * oil_penalty
                + 4.0 * thickness_penalty
                + 1.2 * environmental_penalty
                + 0.5 * abs(pretreatment_time - 5) / 4
            ) + noise
            quality = float(_clip(quality, 0, 100))

            crispness = float(_clip(quality + rng.normal(0, 4), 0, 100))
            color_consistency = float(_clip(quality + rng.normal(0, 5), 0, 100))
            texture = float(_clip(quality + rng.normal(0, 4), 0, 100))
            residual_moisture = float(_clip(
                (raw * 0.12) + (humidity - 55) * 0.035
                + (slice_thickness - 1.4) * 1.8
                - (process_time / 100) * (0.7 if process_name == "Frying" else 0.45)
                + rng.normal(0, 1.2), 0, 20
            ))

            rows.append({
                "batch_id": f"B{batch_no:05d}",
                "product_type": product,
                "process_type": process_name,
                "ambient_temperature_c": round(ambient_temp, 2),
                "relative_humidity_pct": round(humidity, 2),
                "raw_material_moisture_pct": round(raw, 2),
                "slice_thickness_mm": round(slice_thickness, 2),
                "process_temperature_c": round(process_temp, 2),
                "process_time_sec": round(process_time, 2),
                "oil_quality_pct": round(float(oil_quality), 2) if not np.isnan(oil_quality) else np.nan,
                "pretreatment_time_min": round(pretreatment_time, 2),
                "line_speed_m_per_s": round(line_speed, 2),
                "motor_vibration_mm_s": round(motor_vibration, 2),
                "machine_efficiency_pct": round(machine_efficiency, 2),
                "energy_kwh_per_batch": round(float(energy_kwh), 3),
                "residual_moisture_pct": round(residual_moisture, 2),
                "crispness_score": round(crispness, 2),
                "color_consistency_score": round(color_consistency, 2),
                "texture_score": round(texture, 2),
                "quality_score": round(quality, 2),
                "quality_label": "Pass" if quality >= 75 else "Review",
            })
            batch_no += 1

    return pd.DataFrame(rows)

def save_dataset(path: str = "data/synthetic_batches.csv", n_per_product: int = 500, seed: int = 42):
    from pathlib import Path
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_batches(n_per_product=n_per_product, seed=seed)
    df.to_csv(output, index=False)
    return df
