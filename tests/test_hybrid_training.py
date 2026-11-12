import pandas as pd
import pytest
from src.hybrid_training import compatible_columns, run_hybrid_regression

def test_compatible_columns_requires_shared_numeric_variation():
    real = pd.DataFrame({'temp_c':[100,110,120], 'quality':[1.0,2.0,3.0], 'real_only':[1,2,3]})
    synth = pd.DataFrame({'temp_c':[101,111,121], 'quality':[1.1,2.1,3.1], 'synthetic_only':[2,3,4]})
    shared, targets = compatible_columns(real, synth)
    assert shared == ['temp_c','quality']
    assert set(targets) == {'temp_c','quality'}

def test_requires_explicit_compatibility_attestation():
    real = pd.DataFrame({'x':range(20), 'y':range(20)})
    synth = pd.DataFrame({'x':range(20), 'y':range(20)})
    with pytest.raises(ValueError, match='Confirm'):
        run_hybrid_regression(real, synth, 'y', ['x'], compatibility_confirmed=False)

def test_hybrid_model_evaluates_on_real_holdout(tmp_path, monkeypatch):
    import src.hybrid_training as ht
    monkeypatch.setattr(ht, 'OUT', tmp_path / 'models')
    monkeypatch.setattr(ht, 'REPORTS', tmp_path / 'reports')
    real = pd.DataFrame({'x':range(60), 'y':[2*i+1 for i in range(60)]})
    synth = pd.DataFrame({'x':range(60), 'y':[2*i+1 for i in range(60)]})
    report, path = run_hybrid_regression(real, synth, 'y', ['x'], compatibility_confirmed=True, random_state=3)
    assert path.exists()
    assert report['rows']['real_test_held_out'] > 0
    assert report['rows']['combined_train'] == report['rows']['real_train'] + report['rows']['synthetic_train']
    assert 'Real-only Random Forest' in [s['model'] for s in report['scores']]
    assert 'Hybrid Random Forest' in [s['model'] for s in report['scores']]
    assert all(s['status'] == 'ok' for s in report['scores'])
