import pandas as pd
from src.real_data_preprocessing import clean_table, inspect_quality, normalize_columns

def test_column_collision():
    assert normalize_columns(['Fat (%)', 'Fat %', '']) == ['fat', 'fat_2', 'unnamed']

def test_default_preserves_data():
    raw = pd.DataFrame({'Fat (%)': [1., None, 1.], 'Color': ['a', 'b', 'a']})
    clean, steps = clean_table(raw)
    assert len(clean) == 3 and clean['fat'].isna().sum() == 1
    assert inspect_quality(raw)['missing_cells'] == 1

def test_explicit_dedup_and_drop():
    raw = pd.DataFrame({'X': [1., 1., None]})
    clean, _ = clean_table(raw, drop_duplicates=True, missing_strategy='drop_rows')
    assert len(clean) == 1
