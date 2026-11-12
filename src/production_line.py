"""Dynamic discrete-event virtual production line using SimPy.

Process settings now influence thermal-stage duration, pretreatment, slicing,
conveyor/packaging pace, stage temperatures, and estimated energy. All
relationships and coefficients are illustrative assumptions, not calibrated
food-manufacturing physics or validated operating recommendations.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Dict, List

import simpy
import pandas as pd

PRODUCTS = ["Fried Potato Chips", "Baked Potato Chips", "Banana Chips"]

COMMON_STAGES = [
    ("Washing", 1.5, 1),
    ("Peeling", 1.2, 1),
    ("Slicing", 1.0, 1),
    ("Pre-treatment", 1.8, 1),
    ("Drying", 2.5, 1),
]
FRIED_STAGES = [
    ("Frying", 3.0, 1),
    ("Cooling", 1.4, 1),
    ("Seasoning", 0.8, 1),
    ("Quality Inspection", 0.7, 1),
    ("Packaging", 1.0, 1),
]
BAKED_STAGES = [
    ("Baking", 4.0, 1),
    ("Cooling", 1.4, 1),
    ("Seasoning", 0.8, 1),
    ("Quality Inspection", 0.7, 1),
    ("Packaging", 1.0, 1),
]

POWER_KW = {
    "Washing": 1.2, "Peeling": 0.8, "Slicing": 0.7,
    "Pre-treatment": 1.0, "Drying": 2.5, "Frying": 4.0,
    "Baking": 3.5, "Cooling": 1.0, "Seasoning": 0.5,
    "Quality Inspection": 0.3, "Packaging": 0.9,
}

@dataclass
class StageEvent:
    batch_id: str
    product_type: str
    stage: str
    queue_wait_min: float
    process_time_min: float
    start_time_min: float
    end_time_min: float
    machine_id: str
    machine_state: str
    sensor_temperature_c: float
    sensor_humidity_pct: float
    sensor_vibration_mm_s: float
    energy_kwh: float
    applied_temperature_c: float
    setting_effect: str

def _stages_for(product: str):
    if product == "Fried Potato Chips":
        return COMMON_STAGES + FRIED_STAGES
    if product == "Baked Potato Chips":
        return COMMON_STAGES + BAKED_STAGES
    if product == "Banana Chips":
        return [
            ("Washing", 1.2, 1),
            ("Peeling", 0.9, 1),
            ("Slicing", 0.8, 1),
            ("Pre-treatment", 1.2, 1),
            ("Drying", 2.0, 1),
        ] + FRIED_STAGES
    raise ValueError(f"Unsupported product type: {product}")

def _dynamic_stage_parameters(
    stage: str, nominal_time: float, product: str, settings: dict | None
) -> tuple[float, float | None, str, float]:
    """Return illustrative duration, applied temp, effect description, power multiplier."""
    settings = settings or {}
    duration = float(nominal_time)
    applied_temp = None
    effects = []
    power_multiplier = 1.0

    speed = float(settings.get("line_speed_m_per_s", 1.0))
    speed = max(0.1, speed)
    thickness = float(settings.get("slice_thickness_mm", 1.2))
    thickness = max(0.3, thickness)
    pretreat = max(0.0, float(settings.get("pretreatment_time_min", 5.0)))
    temp = float(settings.get("process_temperature_c", 175.0))
    process_seconds = max(1.0, float(settings.get("process_time_sec", 150.0)))

    if stage == "Slicing":
        duration *= max(0.55, min(2.2, thickness / 1.2))
        effects.append("slice thickness")
    elif stage == "Pre-treatment":
        duration *= max(0.25, min(3.0, pretreat / 5.0))
        effects.append("pre-treatment time")
    elif stage in ("Frying", "Baking"):
        baseline_seconds = 450.0 if stage == "Baking" else (120.0 if product == "Banana Chips" else 150.0)
        baseline_temp = 190.0 if stage == "Baking" else (165.0 if product == "Banana Chips" else 175.0)
        time_factor = max(0.35, min(2.5, process_seconds / baseline_seconds))
        temp_factor = max(0.65, min(1.45, baseline_temp / max(temp, 1.0)))
        duration *= time_factor * temp_factor
        applied_temp = temp
        power_multiplier = max(0.7, min(1.5, temp / baseline_temp))
        effects.append("process time and temperature")
    elif stage == "Drying":
        # Thermal setting influences drying load, with a deliberately bounded proxy.
        if product == "Baked Potato Chips":
            drying_reference = 85.0
        elif product == "Banana Chips":
            drying_reference = 80.0
        else:
            drying_reference = 85.0
        # Keep the virtual drying sensor near its own nominal target; the
        # main process-temperature control only changes drying load modestly.
        duration *= max(0.7, min(1.4, drying_reference / max(temp, 1.0)))
        applied_temp = drying_reference
        power_multiplier = max(0.8, min(1.35, temp / drying_reference))
        effects.append("process temperature proxy")
    elif stage in ("Packaging", "Seasoning", "Cooling"):
        duration *= max(0.5, min(1.5, 1.0 / speed))
        effects.append("line speed")
    else:
        # Line speed also modestly affects general handling time.
        if stage in ("Washing", "Peeling"):
            duration *= max(0.7, min(1.3, 1.0 / speed))
            effects.append("line speed")

    return max(0.2, duration), applied_temp, ", ".join(effects) if effects else "nominal", power_multiplier

def simulate_production(
    products: List[str] | None = None,
    batches_per_product: int = 5,
    seed: int = 42,
    machine_capacity: int = 1,
    process_settings: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the virtual line and return stage events, batch summary, and KPIs.

    `process_settings` supports process_temperature_c, process_time_sec,
    slice_thickness_mm, pretreatment_time_min, and line_speed_m_per_s.
    """
    if products is None:
        products = PRODUCTS
    if not products:
        return pd.DataFrame(), pd.DataFrame(), {}

    settings = dict(process_settings or {})
    random.seed(seed)
    env = simpy.Environment()
    machine_capacity = max(1, int(machine_capacity))
    machine_pool: Dict[str, simpy.Resource] = {}
    machine_stats: Dict[str, dict] = {}
    events: List[StageEvent] = []
    batch_summary: List[dict] = []
    batch_counter = 0

    def get_machine(stage: str):
        if stage not in machine_pool:
            machine_pool[stage] = simpy.Resource(env, capacity=machine_capacity)
            machine_stats[stage] = {
                "busy_min": 0.0, "jobs": 0, "queue_wait_min": 0.0, "energy_kwh": 0.0,
            }
        return machine_pool[stage]

    def process_batch(batch_id: str, product: str, release_time: float):
        yield env.timeout(release_time)
        batch_start = env.now
        batch_energy = 0.0
        for stage, nominal_time, _ in _stages_for(product):
            machine = get_machine(stage)
            queue_arrival = env.now
            with machine.request() as request:
                yield request
                start = env.now
                wait = start - queue_arrival
                dynamic_duration, applied_temp, effect, power_multiplier = _dynamic_stage_parameters(
                    stage, nominal_time, product, settings
                )
                # Variability is deterministic for a given seed, settings and run order.
                duration = max(0.2, random.gauss(dynamic_duration, dynamic_duration * 0.08))
                temp = 24 + random.random() * 8
                if stage == "Frying":
                    temp = random.gauss(applied_temp if applied_temp is not None else (175 if product == "Fried Potato Chips" else 165), 2.5)
                elif stage == "Baking":
                    temp = random.gauss(applied_temp if applied_temp is not None else 190, 3.5)
                elif stage == "Drying":
                    temp = random.gauss(applied_temp if applied_temp is not None else 85, 3.0)
                humidity = random.uniform(35, 80)
                vibration = max(0.2, random.gauss(2.8, 0.55))
                power_kw = POWER_KW.get(stage, 1.0) * power_multiplier
                energy = power_kw * duration / 60.0
                yield env.timeout(duration)
                end = env.now
                stats = machine_stats[stage]
                stats["busy_min"] += duration
                stats["jobs"] += 1
                stats["queue_wait_min"] += wait
                stats["energy_kwh"] += energy
                batch_energy += energy
                events.append(StageEvent(
                    batch_id=batch_id, product_type=product, stage=stage,
                    queue_wait_min=round(wait, 3), process_time_min=round(duration, 3),
                    start_time_min=round(start, 3), end_time_min=round(end, 3),
                    machine_id=f"M-{stage.upper().replace(' ', '-')}", machine_state="BUSY",
                    sensor_temperature_c=round(temp, 2),
                    sensor_humidity_pct=round(humidity, 2),
                    sensor_vibration_mm_s=round(vibration, 2),
                    energy_kwh=round(energy, 4),
                    applied_temperature_c=round(applied_temp, 2) if applied_temp is not None else round(temp, 2),
                    setting_effect=effect,
                ))
        batch_summary.append({
            "batch_id": batch_id, "product_type": product,
            "release_time_min": round(release_time, 3),
            "start_time_min": round(batch_start, 3),
            "completion_time_min": round(env.now, 3),
            "lead_time_min": round(env.now - release_time, 3),
            "energy_kwh": round(batch_energy, 4), "status": "Completed",
            "applied_process_temperature_c": settings.get("process_temperature_c"),
            "applied_process_time_sec": settings.get("process_time_sec"),
            "applied_slice_thickness_mm": settings.get("slice_thickness_mm"),
            "applied_pretreatment_time_min": settings.get("pretreatment_time_min"),
            "applied_line_speed_m_per_s": settings.get("line_speed_m_per_s"),
        })

    for product in products:
        for _ in range(max(1, int(batches_per_product))):
            batch_counter += 1
            release = (batch_counter - 1) * 0.8
            env.process(process_batch(f"SIM{batch_counter:05d}", product, release))

    env.run()
    stage_df = pd.DataFrame(asdict(event) for event in events)
    summary_df = pd.DataFrame(batch_summary)
    total_sim_time = float(summary_df["completion_time_min"].max()) if not summary_df.empty else 0.0

    utilization_rows = []
    for stage, stats in machine_stats.items():
        available = total_sim_time * machine_capacity
        utilization = (stats["busy_min"] / available * 100.0) if available > 0 else 0.0
        utilization_rows.append({
            "stage": stage, "jobs_completed": stats["jobs"],
            "total_busy_min": round(stats["busy_min"], 2),
            "average_queue_wait_min": round(stats["queue_wait_min"] / max(stats["jobs"], 1), 3),
            "energy_kwh": round(stats["energy_kwh"], 3),
            "utilization_pct": round(min(100.0, utilization), 2),
            "machine_capacity": machine_capacity,
        })
    utilization_df = pd.DataFrame(utilization_rows).sort_values("utilization_pct", ascending=False)
    kpis = {
        "batches_completed": int(len(summary_df)), "stages_processed": int(len(stage_df)),
        "simulation_duration_min": round(total_sim_time, 2),
        "total_energy_kwh": round(float(summary_df["energy_kwh"].sum()), 3) if not summary_df.empty else 0.0,
        "average_lead_time_min": round(float(summary_df["lead_time_min"].mean()), 2) if not summary_df.empty else 0.0,
        "average_queue_wait_min": round(float(stage_df["queue_wait_min"].mean()), 3) if not stage_df.empty else 0.0,
        "dynamic_settings_applied": bool(settings),
    }
    return stage_df, summary_df, {"kpis": kpis, "machine_utilization": utilization_df}
