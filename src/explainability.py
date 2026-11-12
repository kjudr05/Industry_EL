"""SHAP-based explanations for trained quality prediction models.

The primary implementation explains XGBoost predictions using TreeExplainer.
A model-agnostic fallback is provided for other estimators. Explanations are
relative to a sampled background set and should not be interpreted as causality.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from src.ml_models import FEATURES

def _processed_feature_names(pipeline):
    prep = pipeline.named_steps["preprocess"]
    try:
        return list(prep.get_feature_names_out())
    except Exception:
        return [f"feature_{i}" for i in range(prep.transform(pd.DataFrame(columns=FEATURES)).shape[1])]

def explain_model(pipeline, reference_df: pd.DataFrame, row_df: pd.DataFrame, max_background: int = 80):
    """Return a SHAP explanation for one prediction.

    Returns dict containing prediction, base value, feature contribution frame,
    transformed feature names, and explanation method.
    """
    preprocessor = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]
    background = reference_df[FEATURES].sample(
        n=min(max_background, len(reference_df)), random_state=17
    )
    X_background = preprocessor.transform(background)
    X_row = preprocessor.transform(row_df[FEATURES])
    feature_names = _processed_feature_names(pipeline)

    method = "SHAP"
    try:
        explainer = shap.TreeExplainer(estimator, data=X_background, feature_perturbation="interventional")
        values = explainer.shap_values(X_row)
        base = explainer.expected_value
        if isinstance(values, list):
            values = values[0]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[0]
        values = values.reshape(-1)
        base_array = np.asarray(base).reshape(-1)
        base_value = float(base_array[0]) if len(base_array) else float(base)
    except Exception:
        # KernelExplainer fallback: slower, so keep background small.
        predict_fn = lambda arr: estimator.predict(np.asarray(arr))
        explainer = shap.KernelExplainer(predict_fn, X_background[:min(25, len(X_background))])
        values = np.asarray(explainer.shap_values(X_row, nsamples=100))
        if values.ndim > 1:
            values = values[0]
        values = values.reshape(-1)
        base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    if len(values) != len(feature_names):
        raise ValueError(
            f"SHAP returned {len(values)} contributions for {len(feature_names)} features."
        )

    contribution_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": values,
        "abs_shap_value": np.abs(values),
    }).sort_values("abs_shap_value", ascending=False).reset_index(drop=True)

    prediction = float(np.asarray(pipeline.predict(row_df[FEATURES])).reshape(-1)[0])
    return {
        "prediction": prediction,
        "base_value": base_value,
        "contributions": contribution_df,
        "method": method,
    }

def explain_global(pipeline, data_df: pd.DataFrame, sample_size: int = 250):
    """Calculate mean absolute SHAP values across a representative sample."""
    preprocessor = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]
    sample = data_df[FEATURES].sample(n=min(sample_size, len(data_df)), random_state=23)
    X = preprocessor.transform(sample)
    names = _processed_feature_names(pipeline)

    try:
        explainer = shap.TreeExplainer(estimator)
        raw_values = explainer.shap_values(X)
        if isinstance(raw_values, list):
            raw_values = raw_values[0]
        # Some SHAP/XGBoost version combinations return object/string-wrapped
        # arrays; normalize explicitly before any numeric aggregation.
        values = np.asarray(raw_values, dtype=np.float64)
        if values.ndim == 3:
            values = values[:, :, 0]
        if values.ndim == 1:
            values = values.reshape(1, -1)
        mean_abs = np.mean(np.abs(values), axis=0)
        if len(mean_abs) != len(names):
            raise ValueError("Mismatch between SHAP values and transformed feature names.")
        method = "Tree SHAP"
    except Exception:
        # Reliable fallback: permutation importance on the complete pipeline.
        # This is global feature importance, not a SHAP explanation.
        from sklearn.inspection import permutation_importance
        raw = data_df.sample(n=min(sample_size, len(data_df)), random_state=23)
        result = permutation_importance(
            pipeline, raw[FEATURES], raw["quality_score"].astype(float),
            n_repeats=3, random_state=23, scoring="neg_mean_absolute_error"
        )
        names = FEATURES
        mean_abs = np.maximum(result.importances_mean, 0.0)
        method = "Permutation importance fallback (SHAP conversion failed)"
    return pd.DataFrame({
        "feature": names,
        "mean_abs_shap": mean_abs,
        "importance_method": method,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
