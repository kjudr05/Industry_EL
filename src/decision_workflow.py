"""Human-in-the-loop approval and audit trail for virtual factory recommendations."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import pandas as pd

AUDIT_PATH = Path("data/decision_audit.jsonl")

def append_decision(record: dict, path: str | Path = AUDIT_PATH) -> dict:
    """Append one manager decision as a JSON Lines audit event."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "decision_id": f"DEC-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **record,
    }
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    return event

def read_decisions(path: str | Path = AUDIT_PATH) -> pd.DataFrame:
    """Read audit log, returning an empty frame when no decisions exist."""
    source = Path(path)
    if not source.exists():
        return pd.DataFrame()
    rows = []
    with source.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Ignore incomplete/corrupt trailing lines rather than crashing UI.
                continue
    return pd.DataFrame(rows)

def compare_outcomes(baseline_quality: float, proposed_quality: float,
                     baseline_energy: float, proposed_energy: float) -> dict:
    return {
        "quality_delta": round(float(proposed_quality) - float(baseline_quality), 3),
        "energy_delta_kwh": round(float(proposed_energy) - float(baseline_energy), 4),
    }
