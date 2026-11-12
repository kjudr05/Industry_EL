import pandas as pd
import plotly.express as px
import streamlit as st

from src.simulator import PRODUCTS, generate_batches
from src.production_line import simulate_production
from src.ml_models import train_models, FEATURES
from src.deep_models import train_ann_models, train_lstm_models
from src.model_registry import load_or_train_all, predict_ensemble
from src.explainability import explain_model, explain_global
from src.optimization import make_baseline_row, run_what_if, optimize_settings
from src.decision_workflow import append_decision, read_decisions, compare_outcomes
from src.integration import build_run_record
from src.validation import run_repeatability, summarize_repeatability, run_one_factor_sensitivity
from src.real_data_acquisition import load_catalog, inventory_files, dataset_status

st.set_page_config(page_title="Virtual Chip Factory", page_icon="🏭", layout="wide")

st.title("🏭 Virtual Smart Chip Factory")
st.caption("Software-only simulation • Fried potato chips • Baked potato chips • Banana chips")

st.info(
    "Research prototype: the data and process times are synthetic illustrative assumptions. "
    "They are not validated food-engineering recipes or real factory measurements."
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15 = st.tabs(["Factory Dashboard", "Production Line Simulation", "Batch Data", "AI Model Training", "XAI / SHAP", "What-if & Optimization", "Human Approval & Audit", "Integration & Validation", "Repeatability & Sensitivity", "Phase 12A: Real Data", "Phase 12B: Cleaning & Validation", "Phase 12C: Real-data Models", "Phase 12D: Hybrid Integration", "Phase 12E: Benchmarking", "Phase 12F: Hybrid Training"])

with tab1:
    with st.sidebar:
        st.header("Dataset settings")
        product_filter = st.multiselect("Products", PRODUCTS, default=PRODUCTS)
        batches_per_product = st.slider("Dataset batches per product", min_value=100, max_value=2000, value=500, step=100)
        seed = st.number_input("Dataset random seed", min_value=0, max_value=999999, value=42, step=1, key="dataset_seed")
        generate = st.button("Generate dataset", type="primary", use_container_width=True)

    if "df" not in st.session_state or generate:
        with st.spinner("Generating synthetic manufacturing batches..."):
            st.session_state.df = generate_batches(int(batches_per_product), int(seed))

    df = st.session_state.df
    view = df[df["product_type"].isin(product_filter)].copy() if product_filter else df.iloc[0:0].copy()
    if view.empty:
        st.warning("Select at least one product from the sidebar.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Synthetic batches", f"{len(view):,}")
        c2.metric("Mean simulated quality", f"{view['quality_score'].mean():.1f}/100")
        c3.metric("Pass rate", f"{(view['quality_label'].eq('Pass').mean() * 100):.1f}%")
        c4.metric("Mean energy", f"{view['energy_kwh_per_batch'].mean():.2f} kWh/batch")

        left, right = st.columns(2)
        with left:
            fig = px.box(view, x="product_type", y="quality_score", color="product_type",
                         title="Simulated quality by product", points=False)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            summary = view.groupby("product_type", as_index=False).agg(
                mean_energy=("energy_kwh_per_batch", "mean"),
                mean_quality=("quality_score", "mean"),
            )
            fig = px.bar(summary, x="product_type", y="mean_energy", color="product_type",
                         title="Average synthetic energy per batch",
                         labels={"mean_energy": "kWh per batch"})
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Virtual sensor readings")
        selected_batch = st.selectbox("Select dataset batch", view["batch_id"].tolist())
        batch = view[view["batch_id"] == selected_batch].iloc[0]
        a, b, c = st.columns(3)
        a.metric("Product", batch["product_type"])
        b.metric("Process", batch["process_type"])
        c.metric("Simulated quality", f"{batch['quality_score']:.1f}/100")
        sensor_cols = [
            "ambient_temperature_c", "relative_humidity_pct", "raw_material_moisture_pct",
            "slice_thickness_mm", "process_temperature_c", "process_time_sec",
            "oil_quality_pct", "motor_vibration_mm_s", "machine_efficiency_pct",
        ]
        labels = {
            "ambient_temperature_c": "Ambient temperature (°C)",
            "relative_humidity_pct": "Relative humidity (%)",
            "raw_material_moisture_pct": "Raw material moisture (%)",
            "slice_thickness_mm": "Slice thickness (mm)",
            "process_temperature_c": "Process temperature (°C)",
            "process_time_sec": "Process time (sec)",
            "oil_quality_pct": "Oil quality (%)",
            "motor_vibration_mm_s": "Motor vibration (mm/s)",
            "machine_efficiency_pct": "Machine efficiency (%)",
        }
        sensor_data = [{"Sensor / variable": labels[col], "Reading": batch[col]}
                       for col in sensor_cols if pd.notna(batch[col])]
        st.dataframe(pd.DataFrame(sensor_data), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Discrete-event production line")
    st.write(
        "This simulation moves batches through virtual machines, creates queueing when a machine is busy, "
        "records virtual sensor events, and estimates illustrative machine utilization and energy use."
    )
    with st.form("line_simulation_form"):
        selected_products = st.multiselect("Products to simulate", PRODUCTS, default=PRODUCTS, key="line_products")
        n_batches = st.slider("Batches per selected product", 1, 30, 5)
        capacity = st.slider("Parallel machine capacity per stage", 1, 4, 1)
        line_seed = st.number_input("Line simulation seed", min_value=0, max_value=999999, value=7)
        run_line = st.form_submit_button("Run production line", type="primary")

    if run_line:
        if not selected_products:
            st.warning("Select at least one product.")
        else:
            with st.spinner("Running discrete-event simulation..."):
                events, batches, metrics = simulate_production(
                    products=selected_products,
                    batches_per_product=n_batches,
                    seed=int(line_seed),
                    machine_capacity=capacity,
                )
            st.session_state.line_events = events
            st.session_state.line_batches = batches
            st.session_state.line_metrics = metrics

    if "line_metrics" in st.session_state:
        metrics = st.session_state.line_metrics
        kpis = metrics["kpis"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Completed batches", kpis["batches_completed"])
        k2.metric("Simulation duration", f'{kpis["simulation_duration_min"]:.1f} min')
        k3.metric("Mean lead time", f'{kpis["average_lead_time_min"]:.1f} min')
        k4.metric("Total simulated energy", f'{kpis["total_energy_kwh"]:.2f} kWh')

        utilization = metrics["machine_utilization"]
        left, right = st.columns(2)
        with left:
            fig = px.bar(utilization, x="stage", y="utilization_pct",
                         title="Estimated machine utilization by stage",
                         labels={"stage": "Production stage", "utilization_pct": "Utilization (%)"})
            fig.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.bar(utilization, x="stage", y="average_queue_wait_min",
                         title="Average queue wait by stage",
                         labels={"stage": "Production stage", "average_queue_wait_min": "Wait (min)"})
            fig.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Machine performance summary")
        st.dataframe(utilization, use_container_width=True, hide_index=True)
        st.subheader("Batch completion")
        st.dataframe(st.session_state.line_batches, use_container_width=True, hide_index=True)
        st.subheader("Virtual sensor event log")
        st.dataframe(st.session_state.line_events, use_container_width=True, hide_index=True)
        event_csv = st.session_state.line_events.to_csv(index=False).encode("utf-8")
        batch_csv = st.session_state.line_batches.to_csv(index=False).encode("utf-8")
        d1, d2 = st.columns(2)
        d1.download_button("Download sensor event log", event_csv, "virtual_sensor_events.csv", "text/csv")
        d2.download_button("Download batch completion data", batch_csv, "virtual_batch_completion.csv", "text/csv")
    else:
        st.caption("Configure the run above and select Run production line to generate event logs.")

with tab3:
    st.subheader("Synthetic quality dataset")
    if "df" not in st.session_state:
        st.session_state.df = generate_batches(500, 42)
    df = st.session_state.df
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download synthetic quality dataset",
        df.to_csv(index=False).encode("utf-8"),
        "virtual_chip_factory_batches.csv",
        "text/csv",
    )

st.caption("Phase 2: virtual machine routing, queueing, sensor event logs, machine utilization, and energy estimates.")


with tab4:
    st.subheader("Phase 3 — AI Quality Prediction")
    st.write(
        "Saved models load automatically. On first setup, missing models are trained and saved; subsequent launches reuse those artifacts. Enter cooking/process parameters below to get an ensemble prediction."
    )
    st.warning(
        "The app loads persisted models and trains only when model files are missing. All training labels are synthetic, so metrics do not establish real-world manufacturing accuracy."
    )
    st.markdown("**Models:** Random Forest, XGBoost, ANN (MLP), and LSTM (regression + classification).")
    st.markdown("**Regression metrics:** MAE, RMSE, R². **Classification metrics:** accuracy, macro precision, recall, F1.")

    if "df" not in st.session_state:
        st.session_state.df = generate_batches(500, 42)
    train_df = st.session_state.df

    st.write(f"Available training records: **{len(train_df):,}**")
    st.dataframe(train_df[["product_type", "quality_score", "quality_label"]].groupby(
        ["product_type", "quality_label"], as_index=False
    ).size(), use_container_width=True, hide_index=True)

    # Load persisted models on subsequent launches; train only missing artifacts.
    if "all_model_registry_loaded" not in st.session_state:
        try:
            with st.spinner("Loading saved models; training missing models only on first setup..."):
                registry_metrics, registry_models, registry_split = load_or_train_all(train_df)
            st.session_state.all_model_registry_loaded = True
            st.session_state.ml_metrics = registry_metrics
            st.session_state.ml_models = registry_models
            st.session_state.ml_test_data = registry_split
            st.session_state.ann_metrics = {"regression": [r for r in registry_metrics.get("regression", []) if "ANN" in r.get("model", "")],
                                            "classification": [r for r in registry_metrics.get("classification", []) if "ANN" in r.get("model", "")]}
            st.session_state.lstm_metrics = {"regression": [r for r in registry_metrics.get("regression", []) if "LSTM" in r.get("model", "")],
                                             "classification": [r for r in registry_metrics.get("classification", []) if "LSTM" in r.get("model", "")]}
            if registry_metrics.get("lstm_status"):
                st.warning(registry_metrics["lstm_status"])
        except Exception as exc:
            st.error(f"Model setup failed: {exc}")
            st.info("Check requirements.txt and your Python/TensorFlow compatibility. Models are saved in models/ after successful training.")

    if st.button("Rebuild / retrain all models", key="retrain_all_models"):
        try:
            with st.spinner("Retraining all available models..."):
                registry_metrics, registry_models, registry_split = load_or_train_all(train_df, force_retrain=True)
            st.session_state.all_model_registry_loaded = True
            st.session_state.ml_metrics = registry_metrics
            st.session_state.ml_models = registry_models
            st.session_state.ml_test_data = registry_split
            st.success("Model refresh completed. Saved models will be reused on later launches.")
            st.rerun()
        except Exception as exc:
            st.error(f"Model retraining failed: {exc}")

    if "ml_models" in st.session_state:
        status = st.session_state.get("ml_metrics", {}).get("model_status", {})
        st.markdown("#### Saved model availability")
        st.dataframe(pd.DataFrame([{"model": k, "status": v} for k, v in status.items()]), use_container_width=True, hide_index=True)
        if not st.session_state.get("ml_metrics", {}).get("ensemble_ready", False):
            st.warning("The full four-model ensemble is not ready. Check the LSTM setup message above; the app will use an XGBoost-only fallback for predictions until all four regressors are available.")

    if "ml_metrics" in st.session_state:
        metrics = st.session_state.ml_metrics
        st.markdown("#### Quality score regression")
        reg_df = pd.DataFrame(metrics["regression"])
        st.dataframe(reg_df.round(4), use_container_width=True, hide_index=True)
        fig = px.bar(reg_df, x="model", y="RMSE", title="Regression RMSE (lower is better)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Pass / Review classification")
        cls_df = pd.DataFrame(metrics["classification"])
        st.dataframe(cls_df.round(4), use_container_width=True, hide_index=True)
        fig = px.bar(cls_df, x="model", y="F1_macro", title="Classification macro-F1 (higher is better)")
        st.plotly_chart(fig, use_container_width=True)

        if "ann_metrics" in st.session_state:
            st.markdown("#### ANN (MLP) evaluation")
            st.dataframe(pd.DataFrame(st.session_state.ann_metrics["regression"] + st.session_state.ann_metrics["classification"]).round(4), use_container_width=True, hide_index=True)
        if "lstm_metrics" in st.session_state:
            st.markdown("#### LSTM evaluation")
            st.dataframe(pd.DataFrame(st.session_state.lstm_metrics["regression"] + st.session_state.lstm_metrics["classification"]).round(4), use_container_width=True, hide_index=True)

        st.markdown("#### Predict quality for a custom virtual batch")
        with st.form("custom_batch_form"):
            p = st.selectbox("Product type", PRODUCTS, key="predict_product")
            process = "Baking" if p == "Baked Potato Chips" else "Frying"
            c1, c2, c3 = st.columns(3)
            ambient = c1.number_input("Ambient temperature (°C)", 10.0, 45.0, 27.0)
            humidity = c2.number_input("Relative humidity (%)", 10.0, 100.0, 55.0)
            raw_moisture_default = 63.5 if p == "Banana Chips" else 74.0
            raw_moisture = c3.number_input("Raw material moisture (%)", 30.0, 95.0, raw_moisture_default)
            c4, c5, c6 = st.columns(3)
            thickness = c4.number_input("Slice thickness (mm)", 0.3, 4.0, 1.4)
            temp_default = 190.0 if p == "Baked Potato Chips" else (165.0 if p == "Banana Chips" else 178.0)
            temp = c5.number_input("Process temperature (°C)", 100.0, 240.0, temp_default)
            time_default = 480.0 if p == "Baked Potato Chips" else (125.0 if p == "Banana Chips" else 145.0)
            process_time = c6.number_input("Process time (sec)", 30.0, 700.0, time_default)
            c7, c8, c9 = st.columns(3)
            oil_default = 85.0
            oil_quality = c7.number_input("Oil quality (%)", 0.0, 100.0, oil_default, disabled=(p == "Baked Potato Chips"))
            pretreat = c8.number_input("Pre-treatment time (min)", 0.0, 20.0, 5.0)
            line_speed = c9.number_input("Line speed (m/s)", 0.1, 3.0, 1.0)
            vibration = st.number_input("Motor vibration (mm/s)", 0.1, 10.0, 2.8)
            submit_predict = st.form_submit_button("Predict virtual batch quality")

        if submit_predict:
            model_input = pd.DataFrame([{
                "product_type": p,
                "process_type": process,
                "ambient_temperature_c": ambient,
                "relative_humidity_pct": humidity,
                "raw_material_moisture_pct": raw_moisture,
                "slice_thickness_mm": thickness,
                "process_temperature_c": temp,
                "process_time_sec": process_time,
                "oil_quality_pct": float("nan") if p == "Baked Potato Chips" else oil_quality,
                "pretreatment_time_min": pretreat,
                "line_speed_m_per_s": line_speed,
                "motor_vibration_mm_s": vibration,
                "machine_efficiency_pct": max(50.0, min(100.0, 100 - vibration * 3 - abs(line_speed - 1) * 10)),
                "energy_kwh_per_batch": 1.5 + temp * process_time / (50000 if p == "Baked Potato Chips" else 20000),
            }], columns=FEATURES)
            try:
                predictions = predict_ensemble(model_input, st.session_state.ml_models)
                ensemble_score = predictions["Ensemble"]
                label = "Pass" if ensemble_score >= 75 else "Review"
                r1, r2 = st.columns(2)
                r1.metric("Four-model ensemble quality", f"{ensemble_score:.1f}/100")
                r2.metric("Predicted synthetic batch label", label)
                st.markdown("#### Individual model predictions")
                st.dataframe(
                    pd.DataFrame([{"model": name, "quality_score": score} for name, score in predictions.items()]),
                    use_container_width=True, hide_index=True,
                )
                st.caption("The ensemble is an equal-weight average of Random Forest, XGBoost, ANN and LSTM. Predictions are learned from synthetic labels and are not production recommendations.")
            except Exception as exc:
                st.error(f"Ensemble prediction failed: {exc}")
                if "XGBoost" in st.session_state.ml_models:
                    fallback = float(st.session_state.ml_models["XGBoost"].predict(model_input)[0])
                    st.info(f"XGBoost-only fallback estimate: {max(0, min(100, fallback)):.1f}/100. The four-model ensemble needs all four saved regressors.")


with tab5:
    st.subheader("Phase 4 — Explainable AI (SHAP)")
    st.write(
        "Explain individual quality predictions and compare the features most influential across a sample of batches."
    )
    st.warning(
        "SHAP explains how the trained model uses its inputs relative to a background dataset. "
        "It does not prove causality, and these models currently learn synthetic simulator labels."
    )

    if "df" not in st.session_state:
        st.session_state.df = generate_batches(500, 42)
    xai_df = st.session_state.df

    if "ml_models" not in st.session_state:
        st.info("Train models first in the AI Model Training tab.")
    else:
        model_name = st.selectbox(
            "Model to explain",
            ["XGBoost", "Random Forest"],
            key="xai_model_select"
        )
        available_rows = xai_df.copy()
        chosen_batch = st.selectbox("Choose a batch to explain", available_rows["batch_id"].tolist(), key="xai_batch_select")
        explain_row = available_rows[available_rows["batch_id"] == chosen_batch].iloc[[0]]

        if st.button("Explain selected batch", type="primary", key="explain_one_batch"):
            try:
                with st.spinner("Calculating SHAP contributions..."):
                    explanation = explain_model(
                        st.session_state.ml_models[model_name],
                        xai_df,
                        explain_row,
                    )
                st.session_state.xai_single = explanation
                st.session_state.xai_single_batch = chosen_batch
                st.session_state.xai_single_model = model_name
            except Exception as exc:
                st.error(f"SHAP explanation failed: {exc}")

        if "xai_single" in st.session_state and st.session_state.get("xai_single_batch") == chosen_batch and st.session_state.get("xai_single_model") == model_name:
            explanation = st.session_state.xai_single
            st.metric("Model-predicted quality score", f'{max(0, min(100, explanation["prediction"])):.2f}/100')
            st.caption(f"SHAP baseline: {explanation['base_value']:.2f}. Contributions are relative to this baseline.")
            contrib = explanation["contributions"].head(15).sort_values("shap_value")
            fig = px.bar(
                contrib, x="shap_value", y="feature", orientation="h",
                title="Feature contributions to this prediction",
                labels={"shap_value": "SHAP contribution", "feature": "Input feature"},
                color="shap_value", color_continuous_midpoint=0,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(explanation["contributions"].head(15).round(4), use_container_width=True, hide_index=True)
            st.caption("Positive SHAP values push the model prediction upward; negative values push it downward.")

        st.markdown("#### Global feature importance")
        sample_size = st.slider("Batches to include in global explanation", 50, min(250, max(50, len(xai_df))), min(200, len(xai_df)), 25, key="xai_sample_size")
        if st.button("Calculate global SHAP importance", key="xai_global_button"):
            try:
                with st.spinner("Calculating global SHAP importance..."):
                    global_df = explain_global(
                        st.session_state.ml_models[model_name],
                        xai_df,
                        sample_size=int(sample_size),
                    )
                st.session_state.xai_global = global_df
                st.session_state.xai_global_model = model_name
            except Exception as exc:
                st.error(f"Global SHAP analysis failed: {exc}")

        if "xai_global" in st.session_state and st.session_state.get("xai_global_model") == model_name:
            global_df = st.session_state.xai_global.head(15).sort_values("mean_abs_shap")
            fig = px.bar(
                global_df, x="mean_abs_shap", y="feature", orientation="h",
                title="Global feature importance (mean absolute SHAP)",
                labels={"mean_abs_shap": "Mean |SHAP value|", "feature": "Input feature"},
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(st.session_state.xai_global.round(4), use_container_width=True, hide_index=True)


with tab6:
    st.subheader("Phase 5 — What-if Analysis & Process Optimization")
    st.write(
        "Choose a baseline virtual batch, change process settings, compare the model's estimated quality, "
        "and search a grid of candidate settings."
    )
    st.warning(
        "Recommendations are model-based estimates from synthetic data. The displayed parameter bounds and energy proxy "
        "are illustrative only, not validated food-processing recipes or safe operating limits."
    )

    if "df" not in st.session_state:
        st.session_state.df = generate_batches(500, 42)
    opt_df = st.session_state.df

    if "ml_models" not in st.session_state:
        st.info("Train the models first in the AI Model Training tab.")
    else:
        chosen_product = st.selectbox("Product type", PRODUCTS, key="opt_product")
        product_rows = opt_df[opt_df["product_type"] == chosen_product]
        baseline_id = st.selectbox("Baseline batch", product_rows["batch_id"].tolist(), key="opt_baseline_batch")
        source = product_rows[product_rows["batch_id"] == baseline_id].iloc[0]
        baseline = make_baseline_row(source)

        st.markdown("#### Baseline conditions")
        baseline_cols = [
            "raw_material_moisture_pct", "ambient_temperature_c", "relative_humidity_pct",
            "slice_thickness_mm", "process_temperature_c", "process_time_sec",
            "pretreatment_time_min", "line_speed_m_per_s", "oil_quality_pct",
        ]
        st.dataframe(pd.DataFrame([{
            col: baseline.iloc[0][col] for col in baseline_cols if col in baseline.columns
        }]).T.rename(columns={0: "Baseline value"}), use_container_width=True)

        model = st.session_state.ml_models["XGBoost"]
        base_quality = float(model.predict(baseline)[0])
        base_energy = float(source["energy_kwh_per_batch"])
        b1, b2 = st.columns(2)
        b1.metric("Baseline model quality estimate", f"{max(0, min(100, base_quality)):.2f}/100")
        b2.metric("Baseline synthetic energy estimate", f"{base_energy:.2f} kWh/batch")

        st.markdown("#### What-if scenario")
        with st.form("what_if_form"):
            c1, c2, c3 = st.columns(3)
            temp = c1.number_input(
                "Process temperature (°C)", 100.0, 240.0,
                float(baseline.iloc[0]["process_temperature_c"]), step=1.0
            )
            duration = c2.number_input(
                "Process time (sec)", 30.0, 700.0,
                float(baseline.iloc[0]["process_time_sec"]), step=5.0
            )
            thickness = c3.number_input(
                "Slice thickness (mm)", 0.3, 4.0,
                float(baseline.iloc[0]["slice_thickness_mm"]), step=0.1
            )
            c4, c5 = st.columns(2)
            pretreat = c4.number_input(
                "Pre-treatment time (min)", 0.0, 20.0,
                float(baseline.iloc[0]["pretreatment_time_min"]), step=0.5
            )
            speed = c5.number_input(
                "Line speed (m/s)", 0.1, 3.0,
                float(baseline.iloc[0]["line_speed_m_per_s"]), step=0.1
            )
            run_scenario = st.form_submit_button("Evaluate what-if scenario", type="primary")

        if run_scenario:
            changes = {
                "process_temperature_c": temp,
                "process_time_sec": duration,
                "slice_thickness_mm": thickness,
                "pretreatment_time_min": pretreat,
                "line_speed_m_per_s": speed,
            }
            result = run_what_if(model, baseline, changes)
            st.session_state.what_if_result = result
            st.session_state.what_if_baseline_quality = base_quality
            st.session_state.what_if_baseline_energy = base_energy

        if "what_if_result" in st.session_state:
            result = st.session_state.what_if_result
            q = result["quality_estimate"]
            e = result["energy_estimate_kwh"]
            q_delta = q - st.session_state.what_if_baseline_quality
            e_delta = e - st.session_state.what_if_baseline_energy
            c1, c2, c3 = st.columns(3)
            c1.metric("Scenario quality estimate", f"{q:.2f}/100", delta=f"{q_delta:+.2f} points")
            c2.metric("Scenario energy estimate", f"{e:.2f} kWh/batch", delta=f"{e_delta:+.2f} kWh")
            c3.metric("Quality target status", "Target met" if q >= 85 else "Below target")
            st.caption("Energy is an illustrative proxy; compare trends only until calibrated against measured data.")

        st.markdown("#### Search for candidate settings")
        target = st.slider("Desired quality score", 60, 98, 85, 1, key="opt_target_quality")
        if st.button("Optimize settings", type="primary", key="optimize_settings_button"):
            try:
                with st.spinner("Evaluating candidate parameter combinations..."):
                    results, best = optimize_settings(model, baseline, target_quality=float(target))
                st.session_state.optimization_results = results
                st.session_state.optimization_best = best
                st.session_state.optimization_product = chosen_product
                st.session_state.optimization_target = target
            except Exception as exc:
                st.error(f"Optimization failed: {exc}")

        if "optimization_results" in st.session_state and st.session_state.get("optimization_product") == chosen_product:
            results = st.session_state.optimization_results
            best = st.session_state.optimization_best
            st.markdown("##### Best candidate found by the simulator")
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted quality", f'{best.get("predicted_quality", 0):.2f}/100')
            c2.metric("Estimated energy", f'{best.get("estimated_energy_kwh", 0):.2f} kWh/batch')
            c3.metric("Target status", "Target met" if best.get("target_reached", False) else "Not reached")
            st.dataframe(results.head(20).round(3), use_container_width=True, hide_index=True)
            st.download_button(
                "Download optimization candidates",
                results.to_csv(index=False).encode("utf-8"),
                "virtual_chip_factory_optimization_candidates.csv",
                "text/csv",
            )
            st.caption(
                "The optimizer searches a finite grid; it does not prove a global optimum. "
                "A real deployment would need validated bounds, food-safety constraints, and expert approval."
            )


with tab7:
    st.subheader("Phase 6 — Human-in-the-loop decision workflow")
    st.write(
        "The AI can propose settings, but a floor manager makes the final decision. "
        "Approvals, rejections, and modifications are written to an audit log."
    )
    st.warning(
        "This prototype records decisions and re-runs model estimates. It does not control real equipment. "
        "All quality predictions and energy values are based on synthetic data and illustrative assumptions."
    )

    if "df" not in st.session_state:
        st.session_state.df = generate_batches(500, 42)
    audit_df = st.session_state.df

    if "ml_models" not in st.session_state:
        st.info("Train the models first in the AI Model Training tab.")
    else:
        manager_name = st.text_input("Floor manager / reviewer", value="Demo Manager")
        product = st.selectbox("Product type for decision", PRODUCTS, key="decision_product")
        options = audit_df[audit_df["product_type"] == product]
        batch_id = st.selectbox("Batch to review", options["batch_id"].tolist(), key="decision_batch")
        source = options[options["batch_id"] == batch_id].iloc[0]
        baseline = make_baseline_row(source)
        model = st.session_state.ml_models["XGBoost"]
        baseline_quality = float(max(0, min(100, float(model.predict(baseline)[0]))))
        baseline_energy = float(source["energy_kwh_per_batch"])

        st.markdown("#### Current batch")
        m1, m2, m3 = st.columns(3)
        m1.metric("Baseline quality estimate", f"{baseline_quality:.2f}/100")
        m2.metric("Baseline energy estimate", f"{baseline_energy:.3f} kWh/batch")
        m3.metric("Current simulator label", str(source["quality_label"]))

        st.markdown("#### Proposed settings")
        # Seed controls from latest optimizer result when it belongs to this product;
        # otherwise start from the selected batch's current settings.
        prior_best = st.session_state.get("optimization_best", {})
        use_prior = st.checkbox(
            "Use the latest optimizer candidate when available",
            value=bool(prior_best and st.session_state.get("optimization_product") == product),
            key="decision_use_optimizer"
        )
        defaults = {
            "process_temperature_c": float(prior_best.get("process_temperature_c", source["process_temperature_c"])) if use_prior else float(source["process_temperature_c"]),
            "process_time_sec": float(prior_best.get("process_time_sec", source["process_time_sec"])) if use_prior else float(source["process_time_sec"]),
            "slice_thickness_mm": float(prior_best.get("slice_thickness_mm", source["slice_thickness_mm"])) if use_prior else float(source["slice_thickness_mm"]),
            "pretreatment_time_min": float(prior_best.get("pretreatment_time_min", source["pretreatment_time_min"])) if use_prior else float(source["pretreatment_time_min"]),
            "line_speed_m_per_s": float(prior_best.get("line_speed_m_per_s", source["line_speed_m_per_s"])) if use_prior else float(source["line_speed_m_per_s"]),
        }

        with st.form("decision_proposal_form"):
            c1, c2, c3 = st.columns(3)
            proposed_temp = c1.number_input("Process temperature (°C)", 100.0, 240.0, defaults["process_temperature_c"], step=1.0)
            proposed_time = c2.number_input("Process time (sec)", 30.0, 700.0, defaults["process_time_sec"], step=5.0)
            proposed_thickness = c3.number_input("Slice thickness (mm)", 0.3, 4.0, defaults["slice_thickness_mm"], step=0.1)
            c4, c5 = st.columns(2)
            proposed_pretreat = c4.number_input("Pre-treatment time (min)", 0.0, 20.0, defaults["pretreatment_time_min"], step=0.5)
            proposed_speed = c5.number_input("Line speed (m/s)", 0.1, 3.0, defaults["line_speed_m_per_s"], step=0.1)
            rationale = st.text_area("Manager notes / rationale", placeholder="Why is this change being considered?")
            evaluate_proposal = st.form_submit_button("Evaluate proposed settings")

        if evaluate_proposal:
            changes = {
                "process_temperature_c": proposed_temp,
                "process_time_sec": proposed_time,
                "slice_thickness_mm": proposed_thickness,
                "pretreatment_time_min": proposed_pretreat,
                "line_speed_m_per_s": proposed_speed,
            }
            proposed_result = run_what_if(model, baseline, changes)
            comparison = compare_outcomes(
                baseline_quality, proposed_result["quality_estimate"],
                baseline_energy, proposed_result["energy_estimate_kwh"]
            )
            st.session_state.decision_proposal = {
                "product_type": product,
                "batch_id": batch_id,
                "manager_name": manager_name,
                "baseline": {k: float(baseline.iloc[0][k]) if pd.notna(baseline.iloc[0][k]) else None
                             for k in ["process_temperature_c", "process_time_sec", "slice_thickness_mm", "pretreatment_time_min", "line_speed_m_per_s"]},
                "proposed_settings": changes,
                "baseline_quality_estimate": baseline_quality,
                "proposed_quality_estimate": proposed_result["quality_estimate"],
                "baseline_energy_estimate_kwh": baseline_energy,
                "proposed_energy_estimate_kwh": proposed_result["energy_estimate_kwh"],
                "quality_delta": comparison["quality_delta"],
                "energy_delta_kwh": comparison["energy_delta_kwh"],
                "manager_notes": rationale,
            }

        if "decision_proposal" in st.session_state:
            proposal = st.session_state.decision_proposal
            st.markdown("#### Proposal comparison")
            p1, p2, p3 = st.columns(3)
            p1.metric("Proposed quality estimate", f'{proposal["proposed_quality_estimate"]:.2f}/100',
                      delta=f'{proposal["quality_delta"]:+.2f} points')
            p2.metric("Proposed energy estimate", f'{proposal["proposed_energy_estimate_kwh"]:.3f} kWh/batch',
                      delta=f'{proposal["energy_delta_kwh"]:+.3f} kWh')
            p3.metric("Quality target (demo)", "≥ 85", help="Illustrative threshold only.")
            st.json(proposal["proposed_settings"])

            st.markdown("#### Manager decision")
            d1, d2, d3 = st.columns(3)
            with d1:
                if st.button("Approve proposal", type="primary", key="approve_decision"):
                    event = append_decision({
                        **proposal,
                        "decision": "APPROVED",
                        "decision_by": manager_name,
                        "decision_notes": proposal.get("manager_notes", ""),
                        "simulation_rerun": "Model estimate recomputed; no real equipment controlled",
                    })
                    st.session_state.last_decision = event
                    st.session_state.pop("decision_proposal", None)
                    st.success(f"Decision recorded: {event['decision_id']}")
            with d2:
                if st.button("Reject proposal", key="reject_decision"):
                    event = append_decision({
                        **proposal,
                        "decision": "REJECTED",
                        "decision_by": manager_name,
                        "decision_notes": proposal.get("manager_notes", ""),
                        "simulation_rerun": "Not run; rejected proposal",
                    })
                    st.session_state.last_decision = event
                    st.session_state.pop("decision_proposal", None)
                    st.success(f"Rejection recorded: {event['decision_id']}")
            with d3:
                if st.button("Request changes", key="modify_decision"):
                    event = append_decision({
                        **proposal,
                        "decision": "CHANGES_REQUESTED",
                        "decision_by": manager_name,
                        "decision_notes": proposal.get("manager_notes", ""),
                        "simulation_rerun": "Not run; manager requested changes",
                    })
                    st.session_state.last_decision = event
                    st.session_state.pop("decision_proposal", None)
                    st.success(f"Change request recorded: {event['decision_id']}")

    st.markdown("#### Decision audit trail")
    log_df = read_decisions()
    if log_df.empty:
        st.caption("No decisions recorded yet. Evaluate a proposal and choose an action above.")
    else:
        preferred = [c for c in [
            "decision_id", "timestamp_utc", "product_type", "batch_id",
            "decision_by", "decision", "baseline_quality_estimate",
            "proposed_quality_estimate", "quality_delta",
            "baseline_energy_estimate_kwh", "proposed_energy_estimate_kwh",
            "manager_notes",
        ] if c in log_df.columns]
        st.dataframe(log_df[preferred].iloc[::-1], use_container_width=True, hide_index=True)
        st.download_button(
            "Download decision audit trail",
            log_df.to_csv(index=False).encode("utf-8"),
            "virtual_factory_decision_audit.csv",
            "text/csv",
        )


with tab8:
    st.subheader("Phase 8 — Dynamic process simulation")
    st.write(
        "Apply manager-approved virtual settings to the discrete-event line and inspect how stage timing, "
        "thermal readings, energy proxies, and total lead time respond."
    )
    st.warning(
        "Software simulation only. Settings dynamically influence selected stage durations, thermal-stage readings, "
        "and illustrative energy estimates. These relationships are assumptions—not validated manufacturing physics or safe recipes."
    )

    if "df" not in st.session_state:
        st.session_state.df = generate_batches(500, 42)
    integration_df = st.session_state.df

    if "ml_models" not in st.session_state:
        st.info("Train the models first in the AI Model Training tab.")
    else:
        product = st.selectbox("Product to simulate", PRODUCTS, key="integrated_product")
        product_rows = integration_df[integration_df["product_type"] == product]
        batch_id = st.selectbox("Baseline batch", product_rows["batch_id"].tolist(), key="integrated_batch")
        source = product_rows[product_rows["batch_id"] == batch_id].iloc[0]
        baseline = make_baseline_row(source)

        st.markdown("#### Approved settings to simulate")
        c1, c2, c3 = st.columns(3)
        temp = c1.number_input("Temperature (°C)", 100.0, 240.0, float(source["process_temperature_c"]), key="integrated_temp")
        duration = c2.number_input("Process time (sec)", 30.0, 700.0, float(source["process_time_sec"]), key="integrated_duration")
        thickness = c3.number_input("Slice thickness (mm)", 0.3, 4.0, float(source["slice_thickness_mm"]), key="integrated_thickness")
        c4, c5 = st.columns(2)
        pretreat = c4.number_input("Pre-treatment time (min)", 0.0, 20.0, float(source["pretreatment_time_min"]), key="integrated_pretreat")
        speed = c5.number_input("Line speed (m/s)", 0.1, 3.0, float(source["line_speed_m_per_s"]), key="integrated_speed")
        approved_settings = {
            "process_temperature_c": temp,
            "process_time_sec": duration,
            "slice_thickness_mm": thickness,
            "pretreatment_time_min": pretreat,
            "line_speed_m_per_s": speed,
        }

        manager = st.text_input("Approved by", value="Demo Manager", key="integrated_manager")
        confirmation = st.checkbox("I confirm these are approved for virtual simulation only.", key="integrated_confirm")
        if st.button("Run integrated virtual batch", type="primary", key="run_integrated_batch"):
            if not confirmation:
                st.error("Confirm virtual-simulation-only use before running.")
            else:
                prior = st.session_state.get("integrated_runs", [])
                run_record = build_run_record(
                    batch_id=batch_id,
                    product_type=product,
                    baseline_row=baseline,
                    approved_settings=approved_settings,
                    quality_model=st.session_state.ml_models["XGBoost"],
                    run_number=len(prior) + 1,
                    seed=101 + len(prior),
                )
                run_record["approved_by"] = manager
                st.session_state.integrated_runs = prior + [run_record]
                append_decision({
                    "product_type": product,
                    "batch_id": batch_id,
                    "decision": "APPROVED_SIMULATION_RUN",
                    "decision_by": manager,
                    "proposed_settings": approved_settings,
                    "estimated_quality_score": run_record["estimated_quality_score"],
                    "estimated_energy_kwh": run_record["estimated_energy_kwh"],
                    "run_id": run_record["run_id"],
                    "simulation_rerun": "Integrated virtual batch run completed",
                })
                st.success(f"Virtual run completed: {run_record['run_id']}")

        if "integrated_runs" in st.session_state and st.session_state.integrated_runs:
            run_df = pd.DataFrame(st.session_state.integrated_runs)
            st.markdown("#### Integrated run results")
            latest = st.session_state.integrated_runs[-1]
            a, b, c = st.columns(3)
            a.metric("Latest quality estimate", f'{latest["estimated_quality_score"]:.2f}/100')
            b.metric("Estimated energy", f'{latest["estimated_energy_kwh"]:.3f} kWh/batch')
            c.metric("Simulated line lead time", f'{latest["simulated_line_lead_time_min"]:.2f} min')
            st.dataframe(run_df[[
                "run_id", "product_type", "batch_id", "run_number",
                "estimated_quality_score", "estimated_energy_kwh",
                "simulated_line_lead_time_min", "simulated_line_energy_kwh",
                "dynamic_simulation_settings_applied", "run_status", "approved_by"
            ]], use_container_width=True, hide_index=True)
            st.markdown("#### Latest run: stage-level response")
            stage_events = pd.DataFrame(latest.get("dynamic_stage_events", []))
            if not stage_events.empty:
                st.dataframe(stage_events[[
                    "stage", "process_time_min", "queue_wait_min", "sensor_temperature_c",
                    "energy_kwh", "setting_effect"
                ]], use_container_width=True, hide_index=True)
                st.download_button(
                    "Download latest stage events", stage_events.to_csv(index=False).encode("utf-8"),
                    "dynamic_stage_events.csv", "text/csv",
                )
            st.download_button(
                "Download integrated run history",
                run_df.to_csv(index=False).encode("utf-8"),
                "virtual_factory_integrated_runs.csv",
                "text/csv",
            )

    st.markdown("#### Basic validation")
    st.write("Run the automated tests from the project root using `python -m unittest discover -s tests`.")


with tab9:
    st.subheader("Phase 9 — Repeatability and sensitivity analysis")
    st.write(
        "Run repeatability checks with fixed settings and different random seeds, then vary one setting at a time "
        "to inspect how the synthetic quality and virtual line estimates respond."
    )
    st.warning(
        "This is model-behavior analysis, not real-world validation. Results depend on synthetic data, "
        "the fitted model, and illustrative simulator assumptions."
    )

    if "df" not in st.session_state:
        st.session_state.df = generate_batches(500, 42)
    validation_df = st.session_state.df

    if "ml_models" not in st.session_state:
        st.info("Train the models first in the AI Model Training tab.")
    else:
        v_product = st.selectbox("Product", PRODUCTS, key="validation_product")
        v_rows = validation_df[validation_df["product_type"] == v_product]
        v_batch = st.selectbox("Baseline batch", v_rows["batch_id"].tolist(), key="validation_batch")
        v_source = v_rows[v_rows["batch_id"] == v_batch].iloc[0]
        v_baseline = make_baseline_row(v_source)
        v_settings = {
            "process_temperature_c": float(v_source["process_temperature_c"]),
            "process_time_sec": float(v_source["process_time_sec"]),
            "slice_thickness_mm": float(v_source["slice_thickness_mm"]),
            "pretreatment_time_min": float(v_source["pretreatment_time_min"]),
            "line_speed_m_per_s": float(v_source["line_speed_m_per_s"]),
        }

        st.markdown("#### A. Repeatability experiment")
        rep_count = st.slider("Number of replicate runs", min_value=3, max_value=10, value=5, key="validation_reps")
        if st.button("Run repeatability experiment", type="primary", key="run_repeatability"):
            with st.spinner("Running replicate simulations..."):
                rep_results = run_repeatability(
                    batch_id=v_batch, product_type=v_product, baseline_row=v_baseline,
                    settings=v_settings, quality_model=st.session_state.ml_models["XGBoost"],
                    repetitions=rep_count, seed_start=900,
                )
                st.session_state.validation_repeatability = rep_results
                st.session_state.validation_repeatability_summary = summarize_repeatability(rep_results)

        if "validation_repeatability" in st.session_state:
            rep_results = st.session_state.validation_repeatability
            st.dataframe(st.session_state.validation_repeatability_summary, use_container_width=True, hide_index=True)
            st.dataframe(rep_results, use_container_width=True, hide_index=True)
            st.download_button(
                "Download repeatability results",
                rep_results.to_csv(index=False).encode("utf-8"),
                "repeatability_results.csv", "text/csv",
            )

        st.markdown("#### B. One-factor-at-a-time sensitivity")
        variation_pct = st.slider("Setting variation around baseline (%)", min_value=5, max_value=30, value=10, step=5, key="validation_variation")
        sensitivity_points = st.select_slider("Points per setting", options=[3, 5, 7, 9], value=5, key="validation_sensitivity_points")
        if st.button("Run sensitivity analysis", key="run_sensitivity"):
            with st.spinner("Varying one setting at a time..."):
                sensitivity = run_one_factor_sensitivity(
                    batch_id=v_batch, product_type=v_product, baseline_row=v_baseline,
                    base_settings=v_settings, quality_model=st.session_state.ml_models["XGBoost"],
                    variation_pct=variation_pct, points=sensitivity_points, seed=1200,
                )
                st.session_state.validation_sensitivity = sensitivity

        if "validation_sensitivity" in st.session_state:
            sensitivity = st.session_state.validation_sensitivity
            chosen_metric = st.selectbox(
                "Response to plot",
                ["quality_estimate", "model_energy_estimate_kwh", "line_energy_simulated_kwh", "line_lead_time_min"],
                key="validation_metric",
            )
            selected_setting = st.selectbox("Setting to inspect", sensitivity["setting"].unique().tolist(), key="validation_setting")
            plot_data = sensitivity[sensitivity["setting"] == selected_setting]
            fig = px.line(
                plot_data, x="setting_value", y=chosen_metric, markers=True,
                title=f"{chosen_metric} vs {selected_setting}",
                labels={"setting_value": selected_setting, chosen_metric: chosen_metric.replace("_", " ").title()},
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(sensitivity, use_container_width=True, hide_index=True)
            st.download_button(
                "Download sensitivity results",
                sensitivity.to_csv(index=False).encode("utf-8"),
                "sensitivity_results.csv", "text/csv",
            )

        st.caption(
            "Interpretation tip: sensitivity results show how this synthetic implementation responds to input changes. "
            "They do not establish causation or prove that a real factory would respond in the same way."
        )


with tab10:
    st.header("Phase 12A — Real-data acquisition and provenance")
    st.write("This phase catalogs genuine published measurements and tracks which source files have actually been downloaded. It does not merge them into the synthetic factory dataset yet.")
    catalog = load_catalog()
    for item in catalog["datasets"]:
        with st.expander(item["title"], expanded=False):
            st.markdown(f"**Source:** [{item['source']}]({item['url']})")
            st.markdown(f"**License listed by source:** {item['license']}")
            st.markdown(f"**Format:** {item['format']}")
            st.markdown(f"**Domain:** {item['domain']}")
            st.markdown(f"**Intended role:** {item['role']}")
            st.markdown("**Measured variables:** " + ", ".join(item["measured_variables"]))
            st.markdown("**Limitations:**")
            for limitation in item["limitations"]:
                st.markdown(f"- {limitation}")
            if item.get("download_url"):
                st.markdown(f"[Direct data file]({item['download_url']})")
    st.subheader("Acquisition status")
    status_df = dataset_status()
    st.dataframe(status_df, use_container_width=True, hide_index=True)
    st.caption("A dataset is only marked as locally detected when matching files are present in data/real_data/raw. Catalog entries are not evidence that the files have been downloaded.")
    st.subheader("Inspect locally acquired files")
    inventory = inventory_files()
    if inventory.empty:
        st.info("No downloaded source files detected yet. Use the acquisition instructions below.")
    else:
        st.dataframe(inventory, use_container_width=True, hide_index=True)
        csvs = [x for x in inventory["file"].tolist() if x.lower().endswith(".csv")]
        if csvs:
            selected = st.selectbox("Preview a CSV header", csvs)
            preview_path = __import__("pathlib").Path(__file__).resolve().parent / selected
            try:
                from src.real_data_preprocessing import read_source, sheets as real_sheets
                fmt_sheets = real_sheets(preview_path)
                if fmt_sheets:
                    chosen_sheet = st.selectbox("Workbook sheet", fmt_sheets, key="phase12a_preview_sheet")
                    preview_df = read_source(preview_path, chosen_sheet)
                else:
                    preview_df = read_source(preview_path).head(10)
                st.dataframe(preview_df.head(10), use_container_width=True)
            except Exception as exc:
                st.warning(f"Could not preview this file: {exc}")
                st.caption("The file may be an Excel workbook with a .csv extension, a ZIP archive, or a CSV using a non-UTF-8 encoding. The preview now checks common formats and encodings; ZIP archives must be extracted first.")
    st.subheader("Download instructions")
    st.code("python scripts/download_usda_sweetpotato_dataset.py", language="bash")
    st.markdown("For the two Mendeley datasets, open their source pages above and select **Download All**. Place the original files under `data/real_data/raw/` in folders named `mendeley_frying_oil` and `mendeley_potato_frying_color`.")
    st.warning("Research integrity: these datasets cover different domains and targets. Sweetpotato-chip measurements, potato-frying-oil measurements, French-fry color, and synthetic virtual sensor streams must remain separately identified. We will decide valid feature/target tasks in Phase 12B/12C; do not concatenate rows just because columns look similar.")


with tab11:
    st.header("Phase 12B — Real-data cleaning and validation")
    st.caption("Conservative preprocessing only: preserve raw data, no invented units, labels, or missing values.")
    from src.real_data_preprocessing import source_files, sheets, read_source, profile, process
    files_12b = source_files()
    if not files_12b:
        st.info("No real CSV/Excel files found. Complete Phase 12A acquisition first, then return here.")
    else:
        selected_12b = st.selectbox("Research source file", files_12b, format_func=lambda p: str(p.relative_to(__import__('pathlib').Path(__file__).resolve().parent)))
        sheet_options_12b = sheets(selected_12b)
        sheet_12b = st.selectbox("Excel sheet", sheet_options_12b) if sheet_options_12b else 0
        try:
            source_12b = read_source(selected_12b, sheet_12b)
            summary_12b = profile(source_12b)
            a12, b12, c12 = st.columns(3)
            a12.metric("Source rows", summary_12b["rows"])
            b12.metric("Columns", summary_12b["columns"])
            c12.metric("Exact duplicate rows", summary_12b["duplicate_rows"])
            st.dataframe(source_12b.head(20), use_container_width=True)
            st.subheader("Missingness and type profile")
            st.dataframe(pd.DataFrame(summary_12b["column_profiles"]), use_container_width=True, hide_index=True)
            remove_12b = st.checkbox("Remove exact duplicate rows", value=True)
            if st.button("Generate cleaned CSV and validation report", type="primary"):
                report_12b, report_path_12b = process(selected_12b, sheet_12b, remove_12b)
                st.success("Cleaned dataset and provenance report saved locally.")
                st.code(report_12b["processed_file"])
                st.download_button("Download validation report (JSON)", report_path_12b.read_bytes(), file_name=report_path_12b.name, mime="application/json")
                cleaned_path_12b = __import__('pathlib').Path(__file__).resolve().parent / report_12b["processed_file"]
                st.download_button("Download cleaned CSV", cleaned_path_12b.read_bytes(), file_name=cleaned_path_12b.name, mime="text/csv")
        except Exception as error_12b:
            st.error(f"Unable to read or validate this source: {error_12b}")
    st.warning("This phase does not infer units, fill missing values, remove outliers, or mix real measurements with simulated sensor data. Those decisions require documented domain rules.")


with tab12:
    st.header("Phase 12C — Real-data model baselines")
    st.caption("Train a target-specific baseline on one real research dataset. Synthetic factory rows are never included.")
    from pathlib import Path as _Path12c
    from src.real_data_preprocessing import source_files as _source_files12c, PROCESSED as _PROCESSED12c, read_source as _read_source12c, sheets as _sheets12c
    raw_files_12c = _source_files12c()
    processed_files_12c = sorted(_PROCESSED12c.rglob('*.csv')) if _PROCESSED12c.exists() else []
    choices_12c = [(_Path12c(p), 'raw') for p in raw_files_12c] + [(_Path12c(p), 'cleaned') for p in processed_files_12c]
    if not choices_12c:
        st.info("No real research files detected. Acquire data in Phase 12A, then optionally clean it in Phase 12B.")
    else:
        selected_pair_12c = st.selectbox("Real research dataset", choices_12c, format_func=lambda pair: f"{pair[1].upper()} · {pair[0].name}")
        path_12c, kind_12c = selected_pair_12c
        try:
            if kind_12c == 'raw':
                sheet_choices_12c = _sheets12c(path_12c)
                sheet_choice_12c = st.selectbox("Excel sheet", sheet_choices_12c, key="phase12c_sheet") if sheet_choices_12c else 0
                df_12c = _read_source12c(path_12c, sheet_choice_12c)
            else:
                df_12c = pd.read_csv(path_12c, low_memory=False)
            df_12c.columns = [str(c) for c in df_12c.columns]
            st.write(f"**Rows:** {len(df_12c):,} · **Columns:** {len(df_12c.columns)}")
            st.dataframe(df_12c.head(10), use_container_width=True)
            task_12c = st.radio("Prediction task", ["Regression (measured numeric target)", "Classification (measured categories)"], horizontal=True, key="phase12c_task")
            task_kind_12c = 'regression' if task_12c.startswith('Regression') else 'classification'
            if task_kind_12c == 'regression':
                target_options_12c = [c for c in df_12c.columns if pd.api.types.is_numeric_dtype(df_12c[c]) or pd.to_numeric(df_12c[c], errors='coerce').notna().mean() >= .95]
            else:
                target_options_12c = [c for c in df_12c.columns if 2 <= df_12c[c].nunique(dropna=True) <= 30]
            if not target_options_12c:
                st.warning("No plausible target columns detected for this task. Check the source schema or choose a different dataset/task.")
            else:
                target_12c = st.selectbox("Measured target to predict", target_options_12c, key="phase12c_target")
                feature_options_12c = [c for c in df_12c.columns if c != target_12c]
                numeric_features_12c = [c for c in feature_options_12c if pd.api.types.is_numeric_dtype(df_12c[c])]
                default_features_12c = [c for c in feature_options_12c if not any(term in c.lower() for term in ['id', 'index', 'unnamed'])]
                features_12c = st.multiselect("Input features (exclude IDs, leakage columns, and post-outcome measurements)", feature_options_12c, default=default_features_12c, key="phase12c_features")
                model_options_12c = ["Random Forest", "Extra Trees", "XGBoost", "ANN (MLP)"]
                models_12c = st.multiselect("Baseline models", model_options_12c, default=["Random Forest", "XGBoost", "ANN (MLP)"], key="phase12c_models")
                test_pct_12c = st.slider("Held-out test split (%)", 20, 40, 25, key="phase12c_test_pct")
                seed_12c = st.number_input("Random seed", min_value=0, max_value=999999, value=42, key="phase12c_seed")
                st.warning("Choose features based on the paper's experimental design. Avoid identifiers, data collected after the target outcome, and features that leak the answer. Scores on small laboratory datasets may be unstable.")
                if st.button("Train and evaluate real-data baselines", type="primary", key="phase12c_train"):
                    try:
                        from src.real_data_modeling import train_real_baselines
                        report_12c, report_path_12c, fitted_12c = train_real_baselines(df_12c, target_12c, features_12c, task=task_kind_12c, dataset_name=path_12c.stem, random_state=int(seed_12c), test_size=test_pct_12c / 100, selected_models=models_12c)
                        st.success(f"Completed: {len(fitted_12c)} model(s) trained using real-data rows only.")
                        metric_rows_12c = []
                        for score_12c in report_12c['scores']:
                            metric_rows_12c.append({'Model': score_12c['model'], 'Status': score_12c['status'], **score_12c['metrics'], 'Error': score_12c['error']})
                        st.dataframe(pd.DataFrame(metric_rows_12c), use_container_width=True, hide_index=True)
                        st.caption(f"Training rows: {report_12c['train_rows']} · Held-out rows: {report_12c['test_rows']} · Target: {target_12c}")
                        st.download_button("Download evaluation report (JSON)", report_path_12c.read_bytes(), file_name=report_path_12c.name, mime="application/json")
                        st.download_button("Download model metrics (CSV)", pd.DataFrame(metric_rows_12c).to_csv(index=False).encode(), file_name=f"{path_12c.stem}_metrics.csv", mime="text/csv")
                        st.subheader("Interpretation notes")
                        for note_12c in report_12c['limitations']:
                            st.write(f"• {note_12c}")
                    except Exception as err_12c:
                        st.error(f"Model training could not complete: {err_12c}")
        except Exception as err_12c_load:
            st.error(f"Could not load selected research data: {err_12c_load}")


with tab13:
    st.header("Phase 12D — Hybrid-data integration workspace")
    st.caption("Register and compare real research measurements with synthetic factory records while preserving source provenance.")
    from pathlib import Path as _Path12d
    from src.real_data_preprocessing import source_files as _source_files12d, PROCESSED as _PROCESSED12d, read_source as _read_source12d, sheets as _sheets12d
    from src.hybrid_data_integration import compare_sources as _compare12d, validate_join as _validate_join12d, save_manifest as _save_manifest12d
    raw12d = _source_files12d()
    proc12d = sorted(_PROCESSED12d.rglob("*.csv")) if _PROCESSED12d.exists() else []
    real_choices12d = [(p, "raw real") for p in raw12d] + [(p, "cleaned real") for p in proc12d]
    if not real_choices12d:
        st.info("No research data files found yet. Acquire a real CSV/Excel file in Phase 12A before comparing sources.")
    else:
        selected_real12d = st.selectbox("Real research dataset", real_choices12d, format_func=lambda x: f"{x[1]} · {x[0].name}", key="phase12d_real")
        p12d, kind12d = selected_real12d
        try:
            if kind12d == "raw":
                sheet_opts12d = _sheets12d(p12d)
                sheet12d = st.selectbox("Excel sheet", sheet_opts12d, key="phase12d_sheet") if sheet_opts12d else 0
                real12d = _read_source12d(p12d, sheet12d)
            else:
                real12d = pd.read_csv(p12d, low_memory=False)
            synthetic12d = st.session_state.get("df")
            if synthetic12d is None:
                synthetic12d = generate_batches(500, 42)
                st.session_state.df = synthetic12d
            comparison12d = _compare12d(real12d, synthetic12d)
            st.subheader("Source inventory")
            c1, c2 = st.columns(2)
            c1.metric("Real research rows", f"{len(real12d):,}")
            c1.caption(f"File: {p12d.name} · measured research data")
            c2.metric("Synthetic factory rows", f"{len(synthetic12d):,}")
            c2.caption("Generated by the virtual factory; not real sensor readings")
            st.subheader("Schema comparison")
            st.write("Columns with the same name:", comparison12d["same_named_columns"] or "None")
            st.warning("Matching names are not enough to establish compatible units or scientific meaning. No records are merged automatically.")
            with st.expander("Real data preview"):
                st.dataframe(real12d.head(15), use_container_width=True)
            with st.expander("Synthetic data preview"):
                st.dataframe(synthetic12d.head(15), use_container_width=True)
            st.subheader("Optional join safety check")
            candidate_keys12d = sorted(set(map(str, real12d.columns)) & set(map(str, synthetic12d.columns)))
            if candidate_keys12d:
                key12d = st.selectbox("Candidate shared key (validation only; does not merge rows)", candidate_keys12d, key="phase12d_key")
                valid12d, message12d = _validate_join12d(real12d, synthetic12d, key12d)
                (st.success if valid12d else st.info)(message12d)
                if valid12d:
                    st.warning("A technically valid key is not proof the datasets refer to the same batches. Verify provenance and study design before any join.")
            else:
                st.info("No same-named columns available for a join check. Keeping the sources separate is the correct default.")
            if st.button("Save hybrid integration manifest", type="primary", key="phase12d_save"):
                manifest12d = {**comparison12d, "real_file": str(p12d.relative_to(_Path12d(__file__).resolve().parent)), "real_file_sha256": __import__("hashlib").sha256(p12d.read_bytes()).hexdigest(), "decision": "sources kept separate; no row-level merge performed", "scientific_checks_required": ["Confirm variable definitions and units from the original publication", "Confirm a genuine shared sample or batch identifier before any join", "Do not use synthetic-only features to claim validation on real measured targets without a defensible mapping", "Keep source type in all downstream reports"]}
                manifest_path12d = _save_manifest12d(manifest12d)
                st.success("Integration manifest saved.")
                st.download_button("Download integration manifest (JSON)", manifest_path12d.read_bytes(), file_name=manifest_path12d.name, mime="application/json")
        except Exception as err12d:
            st.error(f"Could not compare the selected data source: {err12d}")
    st.markdown("**Integration rule:** use real-data models for measured quality targets, synthetic data for virtual equipment behavior, and combine them only when an explicit, documented relationship supports it. Scikit-learn pipelines help keep preprocessing fitted on training data rather than leaking information from test data ([documentation](https://scikit-learn.org/stable/common_pitfalls.html)).")


with tab14:
    st.header("Phase 12E — Benchmarking and validation")
    st.caption("Estimate model stability with shuffled cross-validation on one acquired real dataset. Synthetic rows are excluded.")
    from pathlib import Path as _Path12e
    from src.real_data_preprocessing import source_files as _source_files12e, PROCESSED as _PROCESSED12e, read_source as _read_source12e, sheets as _sheets12e
    raw12e = _source_files12e()
    processed12e = sorted(_PROCESSED12e.rglob("*.csv")) if _PROCESSED12e.exists() else []
    choices12e = [(p, "raw") for p in raw12e] + [(p, "cleaned") for p in processed12e]
    if not choices12e:
        st.info("No real dataset files found. Acquire data in Phase 12A before benchmarking.")
    else:
        chosen12e = st.selectbox("Research dataset", choices12e, format_func=lambda x: f"{x[1].upper()} · {x[0].name}", key="phase12e_source")
        path12e, kind12e = chosen12e
        try:
            if kind12e == "raw":
                sheets12e = _sheets12e(path12e)
                sheet12e = st.selectbox("Excel sheet", sheets12e, key="phase12e_sheet") if sheets12e else 0
                df12e = _read_source12e(path12e, sheet12e)
            else: df12e = pd.read_csv(path12e, low_memory=False)
            st.write(f"Rows: {len(df12e):,} · Columns: {len(df12e.columns)}")
            task_label12e = st.radio("Task", ["Regression", "Classification"], horizontal=True, key="phase12e_task")
            task12e = task_label12e.lower()
            if task12e == "regression":
                targets12e = [c for c in df12e.columns if pd.api.types.is_numeric_dtype(df12e[c]) or pd.to_numeric(df12e[c], errors="coerce").notna().mean() >= .95]
            else: targets12e = [c for c in df12e.columns if 2 <= df12e[c].nunique(dropna=True) <= 30]
            if not targets12e: st.warning("No plausible target column found for this task.")
            else:
                target12e = st.selectbox("Measured target", targets12e, key="phase12e_target")
                feat_options12e = [c for c in df12e.columns if c != target12e]
                default12e = [c for c in feat_options12e if not any(t in c.lower() for t in ["id", "index", "unnamed"]) ]
                features12e = st.multiselect("Input features (exclude IDs and leakage columns)", feat_options12e, default=default12e, key="phase12e_features")
                models12e = st.multiselect("Models to compare", ["Random Forest", "Extra Trees", "XGBoost", "ANN (MLP)"], default=["Random Forest", "Extra Trees", "XGBoost", "ANN (MLP)"], key="phase12e_models")
                folds12e = st.slider("Number of CV folds", min_value=2, max_value=10, value=5, key="phase12e_folds")
                seed12e = st.number_input("Random seed", min_value=0, max_value=999999, value=42, key="phase12e_seed")
                st.warning("Use folds appropriate to dataset size. Cross-validation is internal validation, not proof of performance on a different factory or product.")
                if st.button("Run benchmark", type="primary", key="phase12e_run"):
                    try:
                        from src.benchmarking import run_cross_validation
                        report12e, report_path12e = run_cross_validation(df12e, target12e, features12e, task=task12e, dataset_name=path12e.stem, folds=int(folds12e), random_state=int(seed12e), selected_models=models12e)
                        rows12e=[]
                        for result12e in report12e["scores"]:
                            row12e={"Model":result12e["model"],"Status":result12e["status"],"Error":result12e["error"]}
                            for metric12e, value12e in result12e["metrics"].items():
                                row12e[metric12e+" mean"] = value12e["mean"]
                                row12e[metric12e+" std"] = value12e["std"]
                            rows12e.append(row12e)
                        metrics12e=pd.DataFrame(rows12e)
                        st.dataframe(metrics12e, use_container_width=True, hide_index=True)
                        st.caption(f"Target: {target12e} · usable rows: {report12e['usable_rows']} · folds: {report12e['folds']}")
                        st.download_button("Download benchmark report (JSON)", report_path12e.read_bytes(), file_name=report_path12e.name, mime="application/json")
                        st.download_button("Download benchmark metrics (CSV)", metrics12e.to_csv(index=False).encode(), file_name=f"{path12e.stem}_benchmark.csv", mime="text/csv")
                        st.subheader("Interpretation")
                        for note12e in report12e["limitations"]: st.write("• "+note12e)
                    except Exception as err12e: st.error(f"Benchmark failed: {err12e}")
        except Exception as err_load12e: st.error(f"Could not load dataset: {err_load12e}")


with tab15:
    st.header("Phase 12F — Hybrid training with held-out real-data evaluation")
    st.caption("This workflow only trains a combined model when the same target and input features exist in both sources. The real test split is kept completely out of training.")
    from pathlib import Path as _Path12f
    from src.real_data_preprocessing import source_files as _source_files12f, PROCESSED as _PROCESSED12f, read_source as _read_source12f, sheets as _sheets12f
    from src.hybrid_training import compatible_columns as _compatible12f, run_hybrid_regression as _run_hybrid12f
    raw12f = _source_files12f()
    processed12f = sorted(_PROCESSED12f.rglob("*.csv")) if _PROCESSED12f.exists() else []
    choices12f = [(p, "raw") for p in raw12f] + [(p, "cleaned") for p in processed12f]
    if not choices12f:
        st.info("No real research files detected. Acquire a real CSV/Excel dataset in Phase 12A first.")
    else:
        chosen12f = st.selectbox("Real research dataset", choices12f, format_func=lambda x: f"{x[1].upper()} · {x[0].name}", key="phase12f_source")
        path12f, kind12f = chosen12f
        try:
            if kind12f == "raw":
                sheetopts12f = _sheets12f(path12f)
                sheet12f = st.selectbox("Excel sheet", sheetopts12f, key="phase12f_sheet") if sheetopts12f else 0
                real12f = _read_source12f(path12f, sheet12f)
            else:
                real12f = pd.read_csv(path12f, low_memory=False)
            synthetic12f = st.session_state.get("df")
            if synthetic12f is None:
                synthetic12f = generate_batches(500, 42)
                st.session_state.df = synthetic12f
            shared12f, targetopts12f = _compatible12f(real12f, synthetic12f)
            st.metric("Exact shared column names", len(shared12f))
            st.write("Shared columns:", shared12f or "None")
            if not targetopts12f:
                st.warning("No numeric target column with usable variation exists in both this real dataset and the synthetic factory data. A valid combined model cannot be trained on this pair yet. Do not rename unrelated targets to force a match. Choose another real dataset or collect/derive a defensible shared measurement first.")
                st.info("You can still train and benchmark the real-only model in Phase 12C/12E and use synthetic data for virtual-factory testing. This is more scientifically valid than falsely claiming hybrid training.")
            else:
                target12f = st.selectbox("Shared numeric target (must mean the same measured quantity in both sources)", targetopts12f, key="phase12f_target")
                featureopts12f = [c for c in shared12f if c != target12f]
                featuredefaults12f = [c for c in featureopts12f if pd.api.types.is_numeric_dtype(real12f[c]) and pd.api.types.is_numeric_dtype(synthetic12f[c]) and not any(t in c.lower() for t in ["id", "index", "unnamed"])]
                features12f = st.multiselect("Shared input features", featureopts12f, default=featuredefaults12f, key="phase12f_features")
                st.warning("Before proceeding, verify feature and target definitions, units, measurement methods, and product/domain compatibility against the original research source. Matching names alone are insufficient.")
                compatible12f = st.checkbox("I verified that the selected target and features have compatible meanings, units, and measurement conditions in both datasets.", key="phase12f_attest")
                testsize12f = st.slider("Held-out real test fraction", min_value=0.15, max_value=0.40, value=0.25, step=0.05, key="phase12f_testsize")
                seed12f = st.number_input("Random seed", min_value=0, max_value=999999, value=42, key="phase12f_seed")
                if st.button("Train hybrid models and evaluate on held-out real data", type="primary", key="phase12f_train"):
                    try:
                        report12f, reportpath12f = _run_hybrid12f(real12f, synthetic12f, target12f, features12f, compatibility_confirmed=compatible12f, real_source_name=path12f.stem, synthetic_source_name="virtual_factory_batches", random_state=int(seed12f), test_size=float(testsize12f))
                        rows12f=[]
                        for result12f in report12f["scores"]:
                            rows12f.append({"Model": result12f["model"], "Status": result12f["status"], **result12f["metrics"], "Error": result12f["error"]})
                        st.dataframe(pd.DataFrame(rows12f), use_container_width=True, hide_index=True)
                        st.write("Training/evaluation split:", report12f["rows"])
                        st.success("Training complete. Every model was evaluated on the same held-out real test rows.")
                        st.download_button("Download hybrid training report (JSON)", reportpath12f.read_bytes(), file_name=reportpath12f.name, mime="application/json")
                        st.download_button("Download model comparison (CSV)", pd.DataFrame(rows12f).to_csv(index=False).encode(), file_name=f"{path12f.stem}_hybrid_comparison.csv", mime="text/csv")
                        st.subheader("Research interpretation")
                        st.write("Compare Hybrid Random Forest against Real-only Random Forest on the same held-out real rows. A lower MAE/RMSE suggests improvement on this split; it does not by itself prove industrial generalization. Review the report limitations and use grouped/time-aware validation where required.")
                    except Exception as err12f: st.error(f"Hybrid training was not completed: {err12f}")
            with st.expander("Real dataset preview"):
                st.dataframe(real12f.head(10), use_container_width=True)
            with st.expander("Synthetic dataset preview"):
                st.dataframe(synthetic12f.head(10), use_container_width=True)
        except Exception as errload12f: st.error(f"Could not load data for hybrid training: {errload12f}")
