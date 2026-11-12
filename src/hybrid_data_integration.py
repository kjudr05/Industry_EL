"""Phase 12D: provenance-preserving hybrid data workspace.

Real and synthetic records remain separate by default. This module only permits a
row-level join when the user selects an explicit shared key that exists in both
sources and is unique on both sides; it never manufactures links between studies.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HYBRID_ROOT = ROOT / "data" / "hybrid"


def describe_frame(df: pd.DataFrame, source_type: str, source_name: str) -> dict:
    if source_type not in {"real", "synthetic", "derived"}:
        raise ValueError("source_type must be real, synthetic, or derived")
    return {
        "source_name": str(source_name), "source_type": source_type,
        "rows": int(len(df)), "columns": [str(c) for c in df.columns],
        "column_count": int(len(df.columns)),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "content_sha256": hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest(),
    }


def compare_sources(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> dict:
    real_cols = {str(c).strip().lower(): c for c in real_df.columns}
    synth_cols = {str(c).strip().lower(): c for c in synthetic_df.columns}
    common = sorted(set(real_cols) & set(synth_cols))
    return {
        "real": describe_frame(real_df, "real", "selected_research_file"),
        "synthetic": describe_frame(synthetic_df, "synthetic", "virtual_factory_batches"),
        "same_named_columns": common,
        "note": "Same column names do not prove same units, measurement method, population, or scientific meaning.",
        "default_action": "Keep sources separate; compare schemas and only join with a documented shared key.",
    }


def validate_join(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, key: str) -> tuple[bool, str]:
    if not key or key not in real_df.columns or key not in synthetic_df.columns:
        return False, "The selected key must exist in both sources. No rows were joined."
    if real_df[key].isna().any() or synthetic_df[key].isna().any():
        return False, "The join key contains missing values. No rows were joined."
    if real_df[key].duplicated().any() or synthetic_df[key].duplicated().any():
        return False, "The join key is not unique in both sources. No rows were joined to avoid accidental many-to-many matching."
    overlap = set(real_df[key].astype(str)) & set(synthetic_df[key].astype(str))
    if not overlap:
        return False, "The selected key has no matching values. No rows were joined."
    return True, f"Unique shared key validated with {len(overlap)} matching value(s). Confirm scientific validity before joining."


def save_manifest(report: dict) -> Path:
    HYBRID_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = HYBRID_ROOT / f"hybrid_integration_{stamp}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path
