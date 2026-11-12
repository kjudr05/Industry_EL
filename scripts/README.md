# Phase 12A data acquisition

From the project root, with dependencies installed:

```bash
python scripts/download_usda_sweetpotato_dataset.py
```

The download needs internet access on the machine running the project. If blocked, manually download the CSV from the Data.gov dataset page in `data/real_data_catalog.json` and save it under `data/real_data/raw/`.

For the Mendeley datasets, open each source page and choose **Download All**. Preserve the downloaded files and place them under `data/real_data/raw/mendeley_frying_oil/` or `data/real_data/raw/mendeley_potato_frying_color/`. The app intentionally does not guess Mendeley archive internals or create placeholder rows.
