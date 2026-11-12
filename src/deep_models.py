"""ANN and LSTM models for synthetic virtual-factory data."""
from pathlib import Path
import json, warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from src.ml_models import FEATURES, TARGET_REG, TARGET_CLASS

def _preprocessor():
    cat = ["product_type", "process_type"]
    num = [c for c in FEATURES if c not in cat]
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])

def train_ann_models(df, output_dir="models", random_state=42):
    required = set(FEATURES + [TARGET_REG, TARGET_CLASS])
    missing = required - set(df.columns)
    if missing: raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    X = df[FEATURES].copy()
    yr = df[TARGET_REG].astype(float)
    enc = LabelEncoder()
    yc = enc.fit_transform(df[TARGET_CLASS].astype(str))
    Xtr, Xte, yrtr, yrte, yctr, ycte = train_test_split(X, yr, yc, test_size=.2, random_state=random_state, stratify=yc)
    reg = Pipeline([("prep", _preprocessor()), ("model", MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500, early_stopping=True, random_state=random_state, alpha=.001))])
    cls = Pipeline([("prep", _preprocessor()), ("model", MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, early_stopping=True, random_state=random_state, alpha=.001))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg.fit(Xtr, yrtr); cls.fit(Xtr, yctr)
    rp = reg.predict(Xte); cp = cls.predict(Xte).astype(int)
    metrics = {
        "regression": [{"model":"ANN (MLP) Regressor", "MAE":float(mean_absolute_error(yrte,rp)), "RMSE":float(np.sqrt(mean_squared_error(yrte,rp))), "R2":float(r2_score(yrte,rp))}],
        "classification": [{"model":"ANN (MLP) Classifier", "Accuracy":float(accuracy_score(ycte,cp)), "Precision_macro":float(precision_score(ycte,cp,average="macro",zero_division=0)), "Recall_macro":float(recall_score(ycte,cp,average="macro",zero_division=0)), "F1_macro":float(f1_score(ycte,cp,average="macro",zero_division=0))}]
    }
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    joblib.dump(reg,out/"ann_quality_regressor.joblib"); joblib.dump(cls,out/"ann_quality_classifier.joblib"); joblib.dump(enc,out/"ann_label_encoder.joblib")
    return metrics, {"ANN Regressor":reg, "ANN Classifier":cls}

def _make_sequences(df, sequence_length=12, random_state=42, fitted_means=None, fitted_stds=None):
    cols=["process_temperature_c","ambient_temperature_c","relative_humidity_pct","motor_vibration_mm_s","machine_efficiency_pct","energy_kwh_per_batch","raw_material_moisture_pct","line_speed_m_per_s"]
    base=df[cols].astype(float).to_numpy()
    means=np.nanmean(base,axis=0) if fitted_means is None else np.asarray(fitted_means,dtype=float)
    means=np.where(np.isfinite(means),means,0)
    stds=np.nanstd(base,axis=0) if fitted_stds is None else np.asarray(fitted_stds,dtype=float)
    stds=np.where(stds>1e-6,stds,1)
    base=np.where(np.isfinite(base),base,means)
    rng=np.random.default_rng(random_state); seq=[]
    for row in base:
        normalized=(row-means)/stds
        noise=rng.normal(0,.035,size=(sequence_length,len(cols)))
        trend=np.linspace(-.08,.08,sequence_length).reshape(-1,1)
        seq.append(normalized.reshape(1,-1)+noise+trend)
    return np.asarray(seq,dtype=np.float32),cols,means,stds

def train_lstm_models(df, output_dir="models", random_state=42, sequence_length=12, epochs=25):
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Input,LSTM,Dense,Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError as exc:
        raise ImportError("TensorFlow is required for LSTM. Install it with `pip install tensorflow` using a compatible Python version, then restart the app.") from exc
    tf.keras.utils.set_random_seed(random_state)
    X,cols,means,stds=_make_sequences(df,sequence_length,random_state)
    yr=df[TARGET_REG].astype("float32").to_numpy()
    enc=LabelEncoder(); yc=enc.fit_transform(df[TARGET_CLASS].astype(str)).astype("int32")
    idx=np.arange(len(df)); tr,te=train_test_split(idx,test_size=.2,random_state=random_state,stratify=yc)
    callbacks=[EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True)]
    xr=X[tr]; xt=X[te]; yrtr=yr[tr]/100.; yrte=yr[te]; yctr=yc[tr]; ycte=yc[te]
    reg=Sequential([Input(shape=(X.shape[1],X.shape[2])),LSTM(32),Dropout(.15),Dense(16,activation="relu"),Dense(1,activation="sigmoid")])
    reg.compile(optimizer="adam",loss="mse",metrics=["mae"])
    reg.fit(xr,yrtr,validation_split=.15,epochs=max(5,int(epochs)),batch_size=32,callbacks=callbacks,verbose=0)
    rp=reg.predict(xt,verbose=0).reshape(-1)*100.
    cls=Sequential([Input(shape=(X.shape[1],X.shape[2])),LSTM(32),Dropout(.15),Dense(16,activation="relu"),Dense(len(enc.classes_),activation="softmax")])
    cls.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    cls.fit(xr,yctr,validation_split=.15,epochs=max(5,int(epochs)),batch_size=32,callbacks=callbacks,verbose=0)
    cp=np.argmax(cls.predict(xt,verbose=0),axis=1)
    metrics={
      "regression":[{"model":"LSTM Regressor","MAE":float(mean_absolute_error(yrte,rp)),"RMSE":float(np.sqrt(mean_squared_error(yrte,rp))),"R2":float(r2_score(yrte,rp))}],
      "classification":[{"model":"LSTM Classifier","Accuracy":float(accuracy_score(ycte,cp)),"Precision_macro":float(precision_score(ycte,cp,average="macro",zero_division=0)),"Recall_macro":float(recall_score(ycte,cp,average="macro",zero_division=0)),"F1_macro":float(f1_score(ycte,cp,average="macro",zero_division=0))}]
    }
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    reg.save(out/"lstm_quality_regressor.keras"); cls.save(out/"lstm_quality_classifier.keras")
    np.savez_compressed(out/"lstm_sequence_scaling.npz",means=means,stds=stds,feature_columns=np.asarray(cols),label_classes=enc.classes_,sequence_length=np.asarray([sequence_length]))
    with open(out/"lstm_training_metrics.json","w",encoding="utf-8") as f: json.dump(metrics,f,indent=2)
    return metrics,{"LSTM Regressor":reg,"LSTM Classifier":cls}
