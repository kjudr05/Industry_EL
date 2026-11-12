"""Phase 12C: target-specific baselines trained only on acquired real measurements.

The module intentionally does not merge research datasets or synthetic factory rows.
Preprocessing is fitted inside sklearn pipelines after the train/test split.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'models' / 'real_data_baselines'
REPORT_DIR = ROOT / 'data' / 'real_data' / 'model_reports'


def _safe_name(value):
    return re.sub(r'[^a-zA-Z0-9_-]+', '_', str(value)).strip('_')[:80] or 'dataset'


def prepare_task(df, target, features, task='regression'):
    if target not in df.columns:
        raise ValueError('Selected target is not present in the dataset.')
    features = [f for f in features if f in df.columns and f != target]
    if not features:
        raise ValueError('Choose at least one input feature distinct from the target.')
    data = df[features + [target]].copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[target])
    if task == 'regression':
        data[target] = pd.to_numeric(data[target], errors='coerce')
        data = data.dropna(subset=[target])
    else:
        data[target] = data[target].astype('string').str.strip()
        data = data[data[target].notna() & (data[target] != '')]
    if len(data) < 20:
        raise ValueError(f'Only {len(data)} usable rows remain; at least 20 are required for a basic hold-out evaluation.')
    if data[target].nunique() < 2:
        raise ValueError('The selected target has fewer than two distinct values.')
    if task == 'classification' and data[target].nunique() > 30:
        raise ValueError('Classification target has more than 30 classes. Select regression for a continuous measurement.')
    X = data[features].copy()
    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            X[col] = X[col].astype(str)
        # Coerce object columns to numeric only if nearly all non-null values parse.
        if X[col].dtype == 'object' or str(X[col].dtype).startswith('string'):
            parsed = pd.to_numeric(X[col], errors='coerce')
            valid = X[col].notna().sum()
            if valid and parsed.notna().sum() / valid >= .95:
                X[col] = parsed
    y = data[target]
    return X, y, data


def _pipeline(X, estimator, scale_numeric=False):
    numeric = list(X.select_dtypes(include=['number', 'bool']).columns)
    categorical = [c for c in X.columns if c not in numeric]
    numeric_steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale_numeric:
        numeric_steps.append(('scaler', StandardScaler()))
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                         ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    transformers = []
    if numeric:
        transformers.append(('numeric', Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append(('categorical', cat_pipe, categorical))
    return Pipeline([('preprocess', ColumnTransformer(transformers, remainder='drop')),
                     ('model', estimator)])


def train_real_baselines(df, target, features, task='regression', dataset_name='real_dataset', random_state=42, test_size=.25, selected_models=None):
    X, y, data = prepare_task(df, target, features, task)
    class_counts = y.value_counts() if task == 'classification' else None
    stratify = y if task == 'classification' and class_counts.min() >= 2 and len(class_counts) < max(2, int(len(y) * (1-test_size))) else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    candidates = {
        'Random Forest': (RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=random_state, n_jobs=-1) if task == 'regression' else RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight='balanced', random_state=random_state, n_jobs=-1), False),
        'Extra Trees': (ExtraTreesRegressor(n_estimators=250, min_samples_leaf=2, random_state=random_state, n_jobs=-1) if task == 'regression' else ExtraTreesClassifier(n_estimators=250, min_samples_leaf=2, class_weight='balanced', random_state=random_state, n_jobs=-1), False),
        'ANN (MLP)': (MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=700, early_stopping=True, random_state=random_state) if task == 'regression' else MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=700, early_stopping=True, random_state=random_state), True),
    }
    try:
        from xgboost import XGBRegressor, XGBClassifier
        candidates['XGBoost'] = (XGBRegressor(n_estimators=200, max_depth=4, learning_rate=.05, objective='reg:squarederror', random_state=random_state, n_jobs=4) if task == 'regression' else XGBClassifier(n_estimators=200, max_depth=4, learning_rate=.05, eval_metric='mlogloss', random_state=random_state, n_jobs=4), False)
    except Exception:
        pass
    if selected_models:
        candidates = {k:v for k,v in candidates.items() if k in selected_models}
    if not candidates:
        raise ValueError('No available models selected.')
    MODEL_DIR.mkdir(parents=True, exist_ok=True); REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scores, fitted = [], {}
    for name, (estimator, scale) in candidates.items():
        try:
            model = _pipeline(X_train, estimator, scale_numeric=scale)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            if task == 'regression':
                metrics = {'MAE': float(mean_absolute_error(y_test, pred)),
                           'RMSE': float(np.sqrt(mean_squared_error(y_test, pred))),
                           'R2': float(r2_score(y_test, pred)) if len(y_test) > 1 else None}
            else:
                metrics = {'Accuracy': float(accuracy_score(y_test, pred)),
                           'Balanced accuracy': float(balanced_accuracy_score(y_test, pred)),
                           'Precision weighted': float(precision_score(y_test, pred, average='weighted', zero_division=0)),
                           'Recall weighted': float(recall_score(y_test, pred, average='weighted', zero_division=0)),
                           'F1 weighted': float(f1_score(y_test, pred, average='weighted', zero_division=0))}
            model_path = MODEL_DIR / f'{_safe_name(dataset_name)}__{_safe_name(target)}__{_safe_name(name)}.joblib'
            joblib.dump(model, model_path)
            fitted[name] = model
            scores.append({'model': name, 'status': 'ok', 'metrics': metrics, 'model_path': str(model_path.relative_to(ROOT)), 'error': None})
        except Exception as exc:
            scores.append({'model': name, 'status': 'failed', 'metrics': {}, 'model_path': None, 'error': f'{type(exc).__name__}: {exc}'})
    if not fitted:
        raise RuntimeError('All selected models failed: ' + '; '.join(s['model'] + ': ' + str(s['error']) for s in scores))
    report = {
        'phase': '12C', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_name': str(dataset_name), 'source_class': 'externally_sourced_real_data',
        'task': task, 'target': target, 'features': list(X.columns),
        'usable_rows': int(len(data)), 'train_rows': int(len(X_train)), 'test_rows': int(len(X_test)),
        'random_state': int(random_state), 'test_size': float(test_size),
        'preprocessing': 'Median imputation for numeric and most-frequent for categorical features, fitted on training split only within pipeline.',
        'leakage_note': 'Target removed from features; split before fitting imputation/encoding/scaling.',
        'scores': scores,
        'limitations': ['Results apply only to this dataset and selected target/features.', 'Small or laboratory datasets may not generalize to industrial production.', 'No real factory sensor relationship is assumed unless the source data measures it.', 'Synthetic factory rows are not used in this model run.']
    }
    report_path = REPORT_DIR / f'{_safe_name(dataset_name)}__{_safe_name(target)}__report.json'
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    return report, report_path, fitted
