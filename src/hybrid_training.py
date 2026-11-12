"""Phase 12F: guarded real+synthetic training and evaluation on held-out real data.

Only exact shared feature/target columns are eligible. Users must attest that units,
measurement meaning and population are compatible. No implicit label or unit mapping.
"""
from __future__ import annotations
import hashlib, json, re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'models' / 'hybrid_models'
REPORTS = ROOT / 'data' / 'hybrid' / 'reports'

def compatible_columns(real_df: pd.DataFrame, synthetic_df: pd.DataFrame):
    shared = [c for c in real_df.columns if c in synthetic_df.columns]
    numeric_targets = []
    for c in shared:
        r = pd.to_numeric(real_df[c], errors='coerce')
        s = pd.to_numeric(synthetic_df[c], errors='coerce')
        if r.notna().sum() >= 3 and s.notna().sum() >= 3 and r.nunique() > 1 and s.nunique() > 1:
            numeric_targets.append(c)
    return shared, numeric_targets

def _pipeline(X, model, scale=False):
    nums = list(X.select_dtypes(include=['number', 'bool']).columns)
    cats = [c for c in X.columns if c not in nums]
    transformers = []
    if nums:
        steps = [('imputer', SimpleImputer(strategy='median'))]
        if scale: steps.append(('scaler', StandardScaler()))
        transformers.append(('num', Pipeline(steps), nums))
    if cats:
        transformers.append(('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                             ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cats))
    return Pipeline([('preprocess', ColumnTransformer(transformers, remainder='drop')), ('model', model)])

def run_hybrid_regression(real_df, synthetic_df, target, features, *, compatibility_confirmed,
                          real_source_name='real_research_data', synthetic_source_name='virtual_factory_batches', random_state=42, test_size=.25):
    if not compatibility_confirmed:
        raise ValueError('Confirm the feature/target definitions, units, measurement methods, and domain are compatible before training.')
    if target not in real_df or target not in synthetic_df:
        raise ValueError('Target must exist in both real and synthetic datasets with the same meaning and units.')
    features = list(dict.fromkeys(features))
    if not features or any(c not in real_df or c not in synthetic_df for c in features):
        raise ValueError('Every selected feature must exist in both datasets with compatible definitions and units.')
    if target in features: raise ValueError('Target leakage: target cannot be included among input features.')
    if len(real_df) < 12 or len(synthetic_df) < 12:
        raise ValueError('Need at least 12 rows from each source for this guarded demonstration; small studies need a study-specific validation design.')

    # Split real first. The real test partition is never included in fitting.
    real = real_df[features + [target]].copy().replace([np.inf, -np.inf], np.nan)
    real[target] = pd.to_numeric(real[target], errors='coerce')
    real = real.dropna(subset=[target]).reset_index(drop=True)
    synth = synthetic_df[features + [target]].copy().replace([np.inf, -np.inf], np.nan)
    synth[target] = pd.to_numeric(synth[target], errors='coerce')
    synth = synth.dropna(subset=[target]).reset_index(drop=True)
    if len(real) < 12 or len(synth) < 12: raise ValueError('Too few numeric target rows remain after target validation.')
    real_train, real_test = train_test_split(real, test_size=test_size, random_state=random_state)
    # Train rows carry provenance in report; source is not used as a predictive feature.
    train_combined = pd.concat([real_train, synth], ignore_index=True)
    X_train, y_train = train_combined[features], train_combined[target]
    X_test, y_test = real_test[features], real_test[target]
    candidates = {
        'Hybrid Random Forest': (RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=random_state, n_jobs=-1), False),
        'Hybrid Extra Trees': (ExtraTreesRegressor(n_estimators=250, min_samples_leaf=2, random_state=random_state, n_jobs=-1), False),
        'Hybrid ANN (MLP)': (MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=700, early_stopping=True, random_state=random_state), True),
    }
    # Compare against real-only baseline using the exact same held-out real test rows.
    candidates['Real-only Random Forest'] = (RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=random_state, n_jobs=-1), False)
    scores, fitted = [], {}
    for name, (estimator, scale) in candidates.items():
        try:
            train_df = real_train if name.startswith('Real-only') else train_combined
            model = _pipeline(train_df[features], estimator, scale)
            model.fit(train_df[features], train_df[target])
            pred = model.predict(X_test)
            metrics = {'MAE': float(mean_absolute_error(y_test, pred)),
                       'RMSE': float(np.sqrt(mean_squared_error(y_test, pred))),
                       'R2': float(r2_score(y_test, pred)) if len(y_test) > 1 else None}
            fitted[name] = model
            scores.append({'model': name, 'status': 'ok', 'metrics': metrics, 'error': None})
        except Exception as exc:
            scores.append({'model': name, 'status': 'failed', 'metrics': {}, 'error': f'{type(exc).__name__}: {exc}'})
    if not any(s['status'] == 'ok' for s in scores): raise RuntimeError('All hybrid model fits failed.')
    OUT.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)
    safe = lambda x: re.sub(r'[^a-zA-Z0-9_-]+', '_', str(x)).strip('_')[:70] or 'dataset'
    model_paths = {}
    for name, model in fitted.items():
        path = OUT / f'{safe(real_source_name)}__{safe(target)}__{safe(name)}.joblib'
        joblib.dump(model, path)
        try: model_paths[name] = str(path.relative_to(ROOT))
        except ValueError: model_paths[name] = str(path)
    report = {
      'phase': '12F', 'created_utc': datetime.now(timezone.utc).isoformat(), 'task': 'regression',
      'real_source_name': str(real_source_name), 'synthetic_source_name': str(synthetic_source_name),
      'real_source_sha256': hashlib.sha256(real_df.to_csv(index=False).encode()).hexdigest(),
      'synthetic_source_sha256': hashlib.sha256(synthetic_df.to_csv(index=False).encode()).hexdigest(),
      'target': target, 'features': features, 'compatibility_user_attested': True,
      'rows': {'real_total_usable': int(len(real)), 'real_train': int(len(real_train)), 'real_test_held_out': int(len(real_test)), 'synthetic_train': int(len(synth)), 'combined_train': int(len(train_combined))},
      'test_protocol': 'Held-out real rows are split before fitting and never included in any training data. All models scored on the same held-out real rows.',
      'scores': scores, 'model_paths': model_paths,
      'limitations': ['Compatibility checkbox is a user attestation, not independent scientific verification.', 'The synthetic simulator may encode assumed relationships; real-only vs hybrid comparison is required.', 'A random split estimates within-dataset performance and does not prove generalization to another factory/product.', 'Use group/time/site-aware splitting instead of random splitting when samples are repeated by batch, trial, site, or time.', 'No run should be interpreted as real-data validation unless the selected source is genuinely measured and provenance is confirmed.']
    }
    rp = REPORTS / f'{safe(real_source_name)}__{safe(target)}__hybrid_report.json'
    rp.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report, rp
