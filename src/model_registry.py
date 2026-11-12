"""Load persisted models; train and save them only when artifacts are missing."""
from pathlib import Path
import json
import joblib
import pandas as pd
from src.ml_models import train_models
from src.deep_models import train_ann_models, train_lstm_models

MODEL_DIR = Path("models")

def load_or_train_all(dataframe: pd.DataFrame, force_retrain: bool = False):
    """Load saved models on normal launches. Train missing artifacts once."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    baseline_files = {
        "Random Forest": MODEL_DIR / "random_forest_regressor.joblib",
        "XGBoost": MODEL_DIR / "xgboost_regressor.joblib",
    }
    ann_file = MODEL_DIR / "ann_quality_regressor.joblib"
    lstm_file = MODEL_DIR / "lstm_quality_regressor.keras"
    baseline_ready = all(p.exists() for p in baseline_files.values())
    ann_ready = ann_file.exists()
    lstm_ready = lstm_file.exists()

    metrics = {"regression": [], "classification": []}
    models = {}
    split = {}
    if force_retrain or not baseline_ready:
        m, fitted, split = train_models(dataframe, output_dir=str(MODEL_DIR))
        metrics["regression"] += m.get("regression", [])
        metrics["classification"] += m.get("classification", [])
        models.update(fitted)
    else:
        models["Random Forest"] = joblib.load(baseline_files["Random Forest"])
        models["XGBoost"] = joblib.load(baseline_files["XGBoost"])
        # Baseline classifier models may be used by other tabs too.
        for key, file in [
            ("Random Forest Classifier", MODEL_DIR / "random_forest_classifier.joblib"),
            ("XGBoost Classifier", MODEL_DIR / "xgboost_classifier.joblib"),
        ]:
            if file.exists(): models[key] = joblib.load(file)
        if (MODEL_DIR / "training_metrics.json").exists():
            with open(MODEL_DIR / "training_metrics.json", encoding="utf-8") as f:
                saved = json.load(f)
            metrics["regression"] += saved.get("regression", [])
            metrics["classification"] += saved.get("classification", [])

    if force_retrain or not ann_ready:
        am, ann_models = train_ann_models(dataframe, output_dir=str(MODEL_DIR))
        metrics["regression"] += am["regression"]
        metrics["classification"] += am["classification"]
        models.update(ann_models)
    else:
        models["ANN Regressor"] = joblib.load(ann_file)
        cls_file = MODEL_DIR / "ann_quality_classifier.joblib"
        if cls_file.exists(): models["ANN Classifier"] = joblib.load(cls_file)

    if force_retrain or not lstm_ready:
        try:
            lm, lstm_models = train_lstm_models(dataframe, output_dir=str(MODEL_DIR))
            metrics["regression"] += lm["regression"]
            metrics["classification"] += lm["classification"]
            models.update(lstm_models)
        except ImportError as exc:
            # Keep the UI available and make the missing optional runtime explicit.
            metrics["lstm_status"] = str(exc)
    else:
        try:
            import tensorflow as tf
            models["LSTM Regressor"] = tf.keras.models.load_model(lstm_file)
            cls_file = MODEL_DIR / "lstm_quality_classifier.keras"
            if cls_file.exists(): models["LSTM Classifier"] = tf.keras.models.load_model(cls_file)
        except Exception as exc:
            metrics["lstm_status"] = f"Saved LSTM could not be loaded: {exc}"

    metrics["ensemble_ready"] = all(k in models for k in [
        "Random Forest", "XGBoost", "ANN Regressor", "LSTM Regressor"
    ])
    metrics["model_status"] = {
        "Random Forest": "loaded/trained" if "Random Forest" in models else "missing",
        "XGBoost": "loaded/trained" if "XGBoost" in models else "missing",
        "ANN": "loaded/trained" if "ANN Regressor" in models else "missing",
        "LSTM": "loaded/trained" if "LSTM Regressor" in models else "missing",
    }
    return metrics, models, split

def predict_ensemble(input_frame: pd.DataFrame, models: dict):
    """Equal-weight mean of four quality regressors; no hidden re-training."""
    required = ["Random Forest", "XGBoost", "ANN Regressor", "LSTM Regressor"]
    missing = [name for name in required if name not in models]
    if missing:
        raise RuntimeError("Ensemble requires all four trained regressors. Missing: " + ", ".join(missing))
    import numpy as np
    # ANN and XGBoost/RF accept the tabular frame. LSTM expects a synthetic sequence;
    # use the same sequence preparation used in training.
    from src.deep_models import _make_sequences
    scale_path = MODEL_DIR / "lstm_sequence_scaling.npz"
    if scale_path.exists():
        scale = np.load(scale_path, allow_pickle=False)
        seq, _, _, _ = _make_sequences(
            input_frame, sequence_length=12, random_state=42,
            fitted_means=scale["means"], fitted_stds=scale["stds"]
        )
    else:
        seq, _, _, _ = _make_sequences(input_frame, sequence_length=12, random_state=42)
    preds = {
        "Random Forest": float(models["Random Forest"].predict(input_frame)[0]),
        "XGBoost": float(models["XGBoost"].predict(input_frame)[0]),
        "ANN": float(models["ANN Regressor"].predict(input_frame)[0]),
        "LSTM": float(models["LSTM Regressor"].predict(seq, verbose=0).reshape(-1)[0] * 100.0),
    }
    preds = {k: max(0.0, min(100.0, v)) for k, v in preds.items()}
    preds["Ensemble"] = float(np.mean(list(preds.values())))
    return preds
