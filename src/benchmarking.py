"""Phase 12E: repeatable validation and benchmark reports for real-data baselines."""
from __future__ import annotations
import json, re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score, accuracy_score, balanced_accuracy_score, f1_score
from src.real_data_modeling import prepare_task, _pipeline, ROOT
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier

REPORT_DIR = ROOT / 'data' / 'real_data' / 'benchmark_reports'

def _safe(v): return re.sub(r'[^a-zA-Z0-9_-]+', '_', str(v)).strip('_')[:80] or 'dataset'

def run_cross_validation(df, target, features, task='regression', dataset_name='real_dataset', folds=5, random_state=42, selected_models=None):
    X, y, clean = prepare_task(df, target, features, task)
    if task == 'classification':
        counts = y.value_counts()
        folds = min(int(folds), int(counts.min()))
        if folds < 2: raise ValueError('At least two examples per class are required for cross-validation.')
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        scoring = {'accuracy':'accuracy', 'balanced_accuracy':'balanced_accuracy', 'f1_weighted':'f1_weighted'}
        candidates = {
          'Random Forest': (RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight='balanced', random_state=random_state, n_jobs=-1), False),
          'Extra Trees': (ExtraTreesClassifier(n_estimators=250, min_samples_leaf=2, class_weight='balanced', random_state=random_state, n_jobs=-1), False),
          'ANN (MLP)': (MLPClassifier(hidden_layer_sizes=(64,32), max_iter=700, early_stopping=False, random_state=random_state), True)}
    else:
        folds = min(int(folds), len(y))
        if folds < 2: raise ValueError('At least two usable rows are required.')
        splitter = KFold(n_splits=folds, shuffle=True, random_state=random_state)
        scoring = {'mae':'neg_mean_absolute_error', 'rmse':'neg_root_mean_squared_error', 'r2':'r2'}
        candidates = {
          'Random Forest': (RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=random_state, n_jobs=-1), False),
          'Extra Trees': (ExtraTreesRegressor(n_estimators=250, min_samples_leaf=2, random_state=random_state, n_jobs=-1), False),
          'ANN (MLP)': (MLPRegressor(hidden_layer_sizes=(64,32), max_iter=700, early_stopping=False, random_state=random_state), True)}
    try:
        from xgboost import XGBRegressor, XGBClassifier
        if task == 'regression': candidates['XGBoost'] = (XGBRegressor(n_estimators=200, max_depth=4, learning_rate=.05, objective='reg:squarederror', random_state=random_state, n_jobs=4), False)
        else: candidates['XGBoost'] = (XGBClassifier(n_estimators=200, max_depth=4, learning_rate=.05, eval_metric='mlogloss', random_state=random_state, n_jobs=4), False)
    except Exception: pass
    if selected_models: candidates = {k:v for k,v in candidates.items() if k in selected_models}
    results=[]
    for name,(estimator,scale) in candidates.items():
        try:
            model = _pipeline(X, estimator, scale_numeric=scale)
            scores = cross_validate(model, X, y, cv=splitter, scoring=scoring, n_jobs=1, error_score='raise', return_train_score=False)
            metrics={}
            for metric in scoring:
                vals=np.asarray(scores['test_'+metric], dtype=float)
                if metric in ('mae','rmse'): vals=-vals
                metrics[metric]={'mean':float(np.mean(vals)), 'std':float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0, 'folds':[float(v) for v in vals]}
            results.append({'model':name,'status':'ok','metrics':metrics,'error':None})
        except Exception as exc: results.append({'model':name,'status':'failed','metrics':{},'error':f'{type(exc).__name__}: {exc}'})
    if not any(r['status']=='ok' for r in results): raise RuntimeError('All benchmark models failed.')
    report={'phase':'12E','created_utc':datetime.now(timezone.utc).isoformat(),'dataset_name':str(dataset_name),'source_class':'externally_sourced_real_data','task':task,'target':target,'features':list(X.columns),'usable_rows':int(len(clean)),'folds':int(folds),'random_state':int(random_state),'validation':'Shuffled K-fold cross-validation; preprocessing fitted separately within each fold via sklearn Pipeline.','scores':results,'limitations':['Cross-validation estimates performance within this dataset, not external industrial generalization.','Small datasets may have high fold-to-fold variance.','Feature and target selection must follow the source study and avoid leakage.','Synthetic factory records are excluded from these scores.']}
    REPORT_DIR.mkdir(parents=True,exist_ok=True)
    path=REPORT_DIR/f'{_safe(dataset_name)}__{_safe(target)}__cv_report.json'
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report,path
