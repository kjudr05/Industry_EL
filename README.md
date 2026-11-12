# Virtual Chip Factory — Phases 1–7

## Phase 1 — Synthetic batch generator
Product-specific synthetic process, sensor, environmental and quality variables.

## Phase 2 — Discrete-event production line
SimPy routing, machine capacity, queues, virtual sensor events, illustrative energy and utilization.

## Phase 3 — Machine learning
Random Forest and XGBoost regression/classification, held-out metrics, custom batch prediction.

## Phase 4 — Explainable AI
Individual and global SHAP explanations.

## Phase 5 — What-if and optimization
Baseline-vs-scenario comparison, model-predicted quality, illustrative energy proxy, finite-grid candidate search.

## Phase 6 — Human-in-the-loop
Manager review, approval/rejection/change requests, and append-only JSONL audit events.

## Phase 7 — Integration and validation
- Run a model-based quality estimate for approved settings.
- Create an associated discrete-event line run record.
- Track integrated runs in the dashboard and export history.
- Automated tests cover product coverage, quality ranges, and batch completion.

**Integration limitation:** The current SimPy line uses fixed illustrative stage cycle times. Approved settings are recorded and passed to the quality model, but do not yet dynamically change SimPy stage timing or machine energy. That calibration is a future enhancement.

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Run tests:

```powershell
python -m unittest discover -s tests
```

## Research limitations
Data and quality labels are synthetic; model metrics are not real-world performance. Energy values and stage times are illustrative. SHAP explains model attribution rather than causality. Optimization is a finite grid search. Real-world recommendations require measured data, calibrated process models, validated food-safety constraints, and expert review.

## Structure

```text
virtual_chip_factory/
├── app.py
├── requirements.txt
├── README.md
├── data/
├── models/
├── tests/
│   └── test_virtual_factory.py
└── src/
    ├── __init__.py
    ├── simulator.py
    ├── production_line.py
    ├── ml_models.py
    ├── explainability.py
    ├── optimization.py
    ├── decision_workflow.py
    └── integration.py
```


## Phase 8 — Dynamic process simulation

The integration tab now passes approved virtual settings into the SimPy production line. In this phase:
- Process time and temperature influence the illustrative frying/baking stage duration and thermal readings.
- Slice thickness influences the slicing stage duration.
- Pre-treatment time influences the pre-treatment stage duration.
- Line speed influences selected handling, cooling, seasoning, and packaging durations.
- Stage-level event records include process duration, temperature, queue wait, energy proxy, and which setting affected the stage.
- Integrated run history records whether dynamic settings were applied and lets you export the latest stage-event table.

### Run

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open **Integration & Validation / Dynamic process simulation**, train the models first if needed, choose a product and baseline batch, vary one setting at a time, confirm virtual-only use, and run the simulation. Compare the stage-event table and lead-time/energy metrics across runs.

### Scientific limitation

The relationships between settings and stage duration/energy are **illustrative assumptions** designed to make the simulation responsive. They are not calibrated to factory measurements, validated recipes, food-safety limits, or physical equipment. The ML model is trained on synthetic labels, so predicted quality and energy values must not be presented as real-world performance evidence. A later research phase should add sensitivity tests, repeatability experiments, and a clearly documented parameter-calibration strategy.


## Phase 9 — Repeatability and sensitivity analysis

The **Repeatability & Sensitivity** tab adds two experiment types:

1. **Repeatability:** runs the same selected virtual settings across multiple random seeds, then reports mean, standard deviation, minimum, maximum, and coefficient of variation for quality/energy/lead-time estimates.
2. **One-factor-at-a-time sensitivity:** varies each process setting around the selected baseline while holding the other settings fixed. It records the model quality estimate, model energy estimate, simulated line energy, and simulated lead time.

### Recommended workflow

- Train the XGBoost model in **AI Model Training**.
- Choose one product and baseline batch in **Repeatability & Sensitivity**.
- Run 5 replicate runs first and export the CSV.
- Run sensitivity with a 10% variation, inspect one setting at a time, and export the full table.
- Repeat for all three products and compare patterns carefully.

### Interpretation and limitation

Repeatability quantifies variation induced by random seeds in this implementation. Sensitivity describes model response around a selected synthetic baseline. Neither is evidence of external validity, causal effects, food safety, or production-ready process settings. Real-world validation would require measured factory data, domain-expert review, documented experimental design, and calibration of the simulation.


## Phase 9 hotfix — XGBoost class-label encoding

If model training shows `Invalid classes inferred from unique values of y. Expected: [0 1], got ['Pass' 'Review']`, this version fixes it by encoding the string labels with `LabelEncoder` before fitting the classifiers. Metrics are calculated after decoding predictions back to the original `Pass`/`Review` labels. The label encoder is saved as `models/quality_label_encoder.joblib` for consistent future inference.

\n## Phase 10 — ANN and LSTM integration\n\n- **ANN (MLP):** feed-forward neural-network regressor and classifier trained on existing batch-level features.\n- **LSTM:** sequence regressor and classifier. At this phase, sequences are synthetically generated from each batch row, not actual measured sensor time series.\n\nInstall with `pip install -r requirements.txt`, then use the separate ANN and LSTM buttons in the AI Model Training tab. TensorFlow support depends on your Python/OS combination; if installation fails, use a TensorFlow-supported Python version. LSTM training may take longer than the baseline models.\n\n**Research caveat:** all models learn from synthetic targets; the LSTM sequences are illustrative and do not establish real factory performance.\n

## Phase 11 — Persistent four-model ensemble and parameter-driven prediction

- On first launch, the AI tab trains missing model artifacts and saves them under `models/`. On later launches, saved Random Forest, XGBoost, ANN and LSTM models are loaded rather than retrained. Use **Rebuild / retrain all models** only when you intentionally want a refresh.
- The custom batch form accepts process and material parameters. It returns individual quality estimates and an equal-weight four-model ensemble score.
- If TensorFlow/LSTM setup is unavailable, the app clearly reports that the ensemble is incomplete and shows an XGBoost-only fallback instead of falsely claiming a four-model prediction.
- The global XAI routine now converts SHAP outputs explicitly to numeric arrays. If SHAP/XGBoost returns an incompatible object/string representation, it falls back to permutation importance and labels the method honestly.

Model artifacts in `models/` should be retained between launches. Do not delete that folder unless you intend to trigger first-time training again. For reproducibility, keep the same generated training dataset and dependency versions with the saved model files.


## Phase 12A — Real-data acquisition and provenance

Adds a **Phase 12A: Real Data** tab, a source/license/variable catalog, local file inventory, CSV header preview, and a downloader for the public USDA sweetpotato-chip dataset. Two Mendeley datasets are documented with manual download instructions because their source pages distribute versioned archives.

### Acquire data

```bash
python scripts/download_usda_sweetpotato_dataset.py
```

For the frying-oil and potato frying-color datasets, use the source links in the Phase 12A tab, choose **Download All**, and preserve the original files in the named subfolders under `data/real_data/raw/`.

### Research integrity rules

- Raw source files are kept separate from generated synthetic batches.
- The USDA dataset is for **sweetpotato chips**, not ordinary potato chips.
- The Mendeley frying-oil dataset measures oil degradation; it is not a finished-chip quality dataset.
- The 2024 dataset measures **French-fry** color and related conditions, not thin-chip production.
- Phase 12A catalogs sources and detects local files; it does not claim a file was downloaded if it is absent. No real-data model is trained in this sub-phase.

## Phase 12B — Real-data cleaning and validation

After Phase 12A downloads, open **Phase 12B: Cleaning & Validation**. Select an original CSV/XLS/XLSX file (and Excel sheet), review the raw preview, missingness, numeric-parse percentages, duplicate count and value ranges. Choose whether to remove exact duplicates, then click **Generate cleaned CSV and validation report**.

Outputs go to `data/real_data/processed/` and `data/real_data/reports/`; raw research files are never overwritten. Every report records the original relative path, SHA-256, before/after column profiles and explicit transformations. Cleaning normalizes column names, trims strings, turns empty strings into missing values and optionally removes exact duplicates. It intentionally does not infer physical units, impute missing measurements, delete outliers, invent labels, or merge real data with synthetic data. For modeling, verify experimental definitions, units, and target columns manually first.

## Phase 12C — Real-data model baselines

Open **Phase 12C: Real-data Models** after acquiring a real CSV/Excel dataset in Phase 12A. Optionally use the cleaned CSV from Phase 12B. Select a measured numeric target for regression or a categorical target for classification, choose input features based on the source paper, select baseline models, and run a held-out evaluation. The app saves model artifacts under `models/real_data_baselines/` and a JSON report under `data/real_data/model_reports/`.

Important: this is a dataset-specific baseline, not yet a hybrid model. It does not combine different research datasets or synthetic factory batches. Keep identifiers and post-outcome/leakage columns out of the feature selection. The split occurs before imputation, encoding and scaling through an sklearn pipeline. See the [scikit-learn pipeline documentation](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) for why preprocessing is fitted within the training pipeline.

## Phase 12D — Hybrid-data integration

The **Phase 12D: Hybrid Integration** tab compares one acquired research dataset with the virtual factory's synthetic batch table, records schema overlap, and saves a provenance manifest. It does not merge rows automatically. Same-named columns are only schema clues; they do not establish equivalent units, measurement methods, or a shared experimental unit. A candidate key check requires a non-null, unique key on both sides and at least one matching value. Even then, the user must verify the scientific join is valid.

Manifest files are saved under `data/hybrid/`. The raw real-data files and synthetic factory data remain separate. Do not claim that virtual sensor features predict measured real-world quality unless a defensible mapping and real held-out evaluation support that claim.

## Phase 12E — Benchmarking and validation

The `Phase 12E: Benchmarking` tab runs shuffled K-fold cross-validation for selected models on a single acquired real dataset. Preprocessing is encapsulated in a scikit-learn pipeline and is fitted independently within each fold. Regression reports MAE, RMSE and R² mean/std across folds; classification reports accuracy, balanced accuracy and weighted F1 mean/std. JSON and CSV reports are saved under `data/real_data/benchmark_reports/` and can be downloaded from the UI.

**Interpretation:** cross-validation estimates within-dataset performance. It does not establish generalization to another factory, product, or sensor system. Synthetic factory records are not included in these real-data scores. Choose features and targets according to the original study design and exclude identifiers, leakage variables, and measurements that occur after the target outcome.
