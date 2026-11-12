"""Phase 12A: real dataset catalog and local-file inventory. No synthetic data is relabeled as real."""
from pathlib import Path
import json
import pandas as pd
from src.real_data_preprocessing import read_source

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "real_data_catalog.json"
RAW_ROOT = PROJECT_ROOT / "data" / "real_data" / "raw"

def load_catalog():
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

def inventory_files():
    """Return local file inventory, including size and basic tabular metadata where readable."""
    rows = []
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    for path in sorted(RAW_ROOT.rglob("*")):
        if not path.is_file() or path.name.lower() == "readme.md":
            continue
        row = {"file": str(path.relative_to(PROJECT_ROOT)), "bytes": path.stat().st_size, "rows": None, "columns": None, "status": "Found; schema not inspected"}
        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path, nrows=5)
                # Count full rows separately only for manageable files; no full load is needed for inventory.
                row["columns"] = ", ".join(map(str, df.columns))
                row["status"] = "CSV header readable"
            elif path.suffix.lower() in {".xlsx", ".xls"}:
                df = pd.read_excel(path, nrows=5)
                row["columns"] = ", ".join(map(str, df.columns))
                row["status"] = "Spreadsheet header readable"
        except Exception as exc:
            row["status"] = f"File found; could not inspect: {type(exc).__name__}"
        rows.append(row)
    return pd.DataFrame(rows, columns=["file", "bytes", "rows", "columns", "status"])

def dataset_status():
    files = inventory_files()
    catalog = load_catalog()
    status = []
    for item in catalog["datasets"]:
        key = item["dataset_id"]
        matched = []
        for p in RAW_ROOT.rglob("*"):
            if p.is_file() and p.name.lower() != "readme.md":
                if key == "usda_sweetpotato_chip_texture_fat" and "sweetpotato" in p.name.lower(): matched.append(p.name)
                elif key == "mendeley_frying_oil" and "frying_oil" in str(p).lower(): matched.append(p.name)
                elif key == "mendeley_potato_frying_color" and "frying_color" in str(p).lower(): matched.append(p.name)
        status.append({"dataset_id": key, "dataset": item["title"], "local_files_detected": len(matched), "files": ", ".join(matched) if matched else "Not acquired yet", "license": item["license"], "source": item["url"]})
    return pd.DataFrame(status)
