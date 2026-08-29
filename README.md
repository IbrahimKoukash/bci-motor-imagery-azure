# CCAMIC Neuro-Decode · BCI Competition IV 2a
An end-to-end motor-imagery EEG analysis pipeline, from raw recording to
cross-subject decoding, delivered as a live web application on Microsoft Azure,
with results published to Microsoft Fabric and Power BI.

Built at the System Engineering Laboratory, Department of Mechanical Engineering,
Kyushu University.

**Live demo:** https://ibrahim-bci-pipeline.azurewebsites.net

---

## What it does

The [BCI Competition IV dataset 2a](https://www.bbci.de/competition/iv/) processes 22-channel EEG from nine people while they imagine one of four movements (left hand, right hand, feet, or tongue). This project takes a raw session and:

1. **Cleans it** — FIR band-pass, average reference, and ICA-based removal of
   eye-movement artefacts.
2. **Epochs it** — cue-locked trials, baseline-corrected, cropped to the
   event-related desynchronisation (ERD) window where the motor signal is
   strongest.
3. **Decodes it** — eight feature extractors × five classifiers, scored with
   leak-safe cross-validation (GroupKFold on trial id) against the 25% chance
   level for four classes.
4. **Compares across subjects** — a bounded sweep over all nine participants,
   with results persisted for dashboarding.

Everything runs behind a four-tab web UI; no local install is needed to try it.

## Try it without installing anything

Open the [live demo](https://ibrahim-bci-pipeline.azurewebsites.net) and go to
the **Cross-subject sweep** tab, nine participants are already preprocessed and
bundled, so you can pick a few, press run, and see results in under a minute.
For the full single-subject pipeline, the **Preprocess** tab ships with an
example recording you can load directly.

## The interface

The app is organised as four tabs. The screenshots below show the controls for
each; the result views (accuracy heatmaps, confusion matrices, ERD topomaps)
appear once a run completes.

**1 · Preprocess** — load a bundled example or upload a `.mat`, set the filter
and motor-imagery window, and run cleaning end to end.

<img width="417" height="532" alt="image" src="https://github.com/user-attachments/assets/979283f8-f013-4ae5-9c0e-298336b0a2b0" />

**2 · Decode & compare** — pick any feature extractors and classifiers, set the
window, and score every pairing.

<img width="469" height="550" alt="image" src="https://github.com/user-attachments/assets/7daabfaf-76cd-4700-a229-178f0a5a2b34" />

**3 · Advanced** — tune per-method parameters; settings are fixed across the
whole comparison so every pairing is judged on the same footing.

<img width="730" height="281" alt="image" src="https://github.com/user-attachments/assets/f547b2bb-69c3-4a0e-bbbf-44ec2baf8bff" />

**4 · Cross-subject sweep** — pick subjects, features and classifiers, and sweep
across the bundled cache with no upload needed.

<img width="383" height="615" alt="image" src="https://github.com/user-attachments/assets/b5195946-5c3a-4ca5-b416-41dd9cae348c" />

## Architecture

| Layer | Service | Role |
|-------|---------|------|
| UI + compute | Azure App Service (Linux container) | Runs the Streamlit app and the MNE pipeline |
| Image | Azure Container Registry | Holds the built container image |
| Persistence | Azure Blob Storage | One Parquet file per sweep run |
| Identity | Microsoft Entra ID | App authenticates to storage via managed identity — no keys in code |
| Analytics | Microsoft Fabric + Power BI | Lakehouse table and cross-subject dashboard |

The app degrades gracefully: if Blob storage is unreachable it still runs and is
fully demoable, results just aren't persisted (`LOCAL_ONLY=1` forces this).

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# get the dataset (.mat files) from the BNCI Horizon 2020 archive:
# http://bnci-horizon-2020.eu/database/data-sets (dataset 001-2014)

# pre-cache subjects once so the sweep tab is fast (optional):
python cache_subjects.py --data-dir path/to/BCICIV-2a-mat

LOCAL_ONLY=1 streamlit run app_streamlit.py
```
`LOCAL_ONLY=1` disables Azure Blob entirely, so you can run with no cloud
credentials.

## Project layout

| File | Purpose |
|------|---------|
| `app_streamlit.py` | Main four-tab UI |
| `pipeline.py` | Load → filter → ICA → epoch → ERD-crop preprocessing |
| `decoding.py` | Windowing, the eight feature extractors, leak-safe pipeline |
| `decoding_algorithms.py` | Registers the five classifiers, confusion-matrix eval |
| `decoding_params.py`, `params.py`, `advanced_ui.py` | Parameter-aware pipeline + Advanced tab |
| `comparison.py` | Feature × classifier grid, writes Fabric-ready Parquet |
| `sweep_tab.py`, `sweep_all_subjects.py` | Cross-subject sweep (in-app and offline batch) |
| `cache_subjects.py` | One-off offline preprocessing cache |
| `storage.py` | Azure Blob access via managed identity |
| `plots.py`, `viz_extra.py`, `flow.py` | Dark-themed figures and the pipeline flow strip |

## Methodology notes

- **Leak safety.** Every learning step (scaler, CSP, classifier) lives in one
  scikit-learn `Pipeline`, re-fit per fold. Windows carry their trial id so
  `GroupKFold` never splits one trial across train and test. Validated with a
  null (label-shuffle) test.
- **CSP** uses Ledoit-Wolf regularisation, since the data is rank-deficient after
  average reference and ICA.
- **Bands.** Mu (8–12 Hz) and beta (12–30 Hz) are the reported bands for motor
  imagery. Higher-frequency claims are avoided here because of EMG-contamination
  risk.

## Data & licence

This project does not redistribute the dataset. BCI Competition IV 2a is
available from the [BNCI Horizon 2020 archive](http://bnci-horizon-2020.eu/database/data-sets)
(dataset 001-2014) under CC BY-ND 4.0. Please cite the original dataset authors
if you use it.

The code in this repository is released under the MIT Licence — see `LICENSE`.

## Author

Ibrahim Koukash — PhD candidate, Mechanical Engineering, Kyushu University.
Research in AI, biosignal processing, and human–robot interaction for assistive
technology.
