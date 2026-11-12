"""Phase 12B: conservative cleaning, profiles, and source provenance."""
from pathlib import Path
import hashlib
import json
import re
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/real_data/raw'
PROCESSED = ROOT / 'data/real_data/processed'
REPORTS = ROOT / 'data/real_data/reports'
EXTENSIONS = {'.csv', '.xlsx', '.xls'}

def source_files():
    RAW.mkdir(parents=True, exist_ok=True)
    return sorted(p for p in RAW.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)

def normalize_columns(columns):
    counts, result = {}, []
    for col in columns:
        base = re.sub(r'[^a-z0-9]+', '_', str(col).lower()).strip('_') or 'unnamed'
        counts[base] = counts.get(base, 0) + 1
        result.append(base if counts[base] == 1 else f'{base}_{counts[base]}')
    return result

def _actual_format(path):
    """Detect common tabular formats from file content, not only the filename extension."""
    path = Path(path)
    try:
        with path.open('rb') as f:
            signature = f.read(8)
        if signature.startswith(b'PK\x03\x04') and zipfile.is_zipfile(path):
            # Excel workbooks are ZIP containers with xl/workbook.xml.
            with zipfile.ZipFile(path) as zf:
                if any(n.startswith('xl/workbook.xml') for n in zf.namelist()):
                    return 'xlsx'
                return 'zip'
        if signature.startswith(bytes.fromhex('D0CF11E0A1B11AE1')):
            return 'xls'
        if path.suffix.lower() in {'.csv', '.txt'}:
            return 'csv'
        if path.suffix.lower() in {'.xlsx', '.xls'}:
            return path.suffix.lower().lstrip('.')
    except OSError:
        pass
    return 'unknown'

def sheets(path):
    fmt = _actual_format(path)
    if fmt in {'xls', 'xlsx'}:
        return pd.ExcelFile(path).sheet_names
    return []

def read_source(path, sheet=0):
    path = Path(path).resolve()
    if not path.is_relative_to(RAW.resolve()) or path.suffix.lower() not in EXTENSIONS:
        raise ValueError('Source must be a supported file in the raw data folder')
    fmt = _actual_format(path)
    if fmt == 'csv':
        last_error = None
        for encoding in ('utf-8-sig', 'utf-8', 'cp1252', 'latin1'):
            try:
                return pd.read_csv(path, encoding=encoding, low_memory=False)
            except UnicodeDecodeError as exc:
                last_error = exc
        raise ValueError(f'Could not decode CSV using common encodings: {last_error}')
    if fmt in {'xls', 'xlsx'}:
        return pd.read_excel(path, sheet_name=sheet)
    if fmt == 'zip':
        raise ValueError('This is a ZIP archive, not a CSV or Excel workbook. Extract it first and select a CSV/XLS/XLSX file inside.')
    raise ValueError('Unrecognized file format. Expected a CSV file or Excel workbook (.xls/.xlsx).')

def profile(df):
    cols = []
    for c in df.columns:
        s = df[c]
        numeric = pd.to_numeric(s, errors='coerce')
        cols.append({'column': str(c), 'dtype': str(s.dtype),
                     'missing_count': int(s.isna().sum()),
                     'missing_pct': round(float(s.isna().mean() * 100), 2) if len(s) else 0,
                     'unique_count': int(s.nunique(dropna=True)),
                     'numeric_parse_pct': round(float(numeric.notna().mean() * 100), 2) if len(s) else 0,
                     'numeric_min': float(numeric.min()) if numeric.notna().any() else None,
                     'numeric_max': float(numeric.max()) if numeric.notna().any() else None})
    return {'rows': len(df), 'columns': len(df.columns),
            'duplicate_rows': int(df.duplicated().sum()), 'column_profiles': cols}

def clean(df, remove_duplicates=True):
    result = df.copy()
    old_cols = list(map(str, result.columns))
    result.columns = normalize_columns(result.columns)
    for col in result.select_dtypes(include=['object', 'string']).columns:
        result[col] = result[col].map(lambda v: v.strip() if isinstance(v, str) else v)
        result[col] = result[col].replace('', float('nan'))
    duplicates = int(result.duplicated().sum())
    if remove_duplicates:
        result = result.drop_duplicates().reset_index(drop=True)
    return result, {'original_columns': old_cols, 'clean_columns': list(result.columns),
                    'duplicates_removed': duplicates if remove_duplicates else 0,
                    'imputed_values': 0, 'outliers_removed': 0, 'unit_conversions': [],
                    'note': 'Units, target meaning and missing-value strategy require domain review.'}

def process(path, sheet=0, remove_duplicates=True):
    path = Path(path).resolve()
    before_df = read_source(path, sheet)
    cleaned, changes = clean(before_df, remove_duplicates)
    relative = path.relative_to(RAW.resolve())
    suffix = f'_sheet_{sheet}' if path.suffix.lower() != '.csv' else ''
    name = re.sub(r'[^a-z0-9]+', '_', path.stem.lower()).strip('_')
    output = PROCESSED / relative.parent / f'{name}{suffix}.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output, index=False)
    report = {'source_file': str(relative), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'source_class': 'externally_sourced_real_data', 'sheet': sheet if suffix else None,
              'processed_file': str(output.relative_to(ROOT)), 'before': profile(before_df),
              'after': profile(cleaned), 'changes': changes,
              'warnings': ['Verify scientific units and variable meanings against the source publication.',
                           'No guessed imputation, unit conversion, target label, or outlier deletion.',
                           'Real and synthetic rows are never merged in this phase.']}
    report_path = REPORTS / f'{hashlib.sha256((str(relative)+suffix).encode()).hexdigest()[:16]}.json'
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    return report, report_path

# Public table helpers used by automated tests and lightweight notebook workflows.
def inspect_quality(df):
    prof = profile(df)
    return {
        'rows': prof['rows'], 'columns': prof['columns'],
        'missing_cells': int(df.isna().sum().sum()),
        'duplicate_rows': prof['duplicate_rows'],
        'column_profiles': prof['column_profiles'],
    }

def clean_table(df, drop_duplicates=False, missing_strategy='preserve'):
    """Clean an in-memory table conservatively; no implicit imputation or unit changes."""
    result, changes = clean(df, remove_duplicates=drop_duplicates)
    if missing_strategy == 'drop_rows':
        before = len(result)
        result = result.dropna().reset_index(drop=True)
        changes['rows_dropped_for_missing_values'] = before - len(result)
    elif missing_strategy != 'preserve':
        raise ValueError("missing_strategy must be 'preserve' or 'drop_rows'")
    return result, changes
