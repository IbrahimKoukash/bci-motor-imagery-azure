"""
Feature x classifier comparison + Fabric-ready result persistence.
"""
import datetime as dt
import uuid

import numpy as np
import pandas as pd

from decoding import evaluate
from storage import upload_dataframe_parquet, BLOB_ENABLED, RESULTS

# Fixed schema -> every Parquet file stacks cleanly in a Fabric Lakehouse.
RESULT_SCHEMA = {
    "run_id": "string", "timestamp_utc": "datetime64[ns]", "user_id": "string",
    "subject": "string", "feature_method": "string", "classifier": "string",
    "window_sec": "float64", "overlap_pct": "float64", "cv_folds": "int64",
    "n_windows": "int64", "n_trials": "int64",
    "mean_accuracy": "float64", "std_accuracy": "float64", "rank": "int64",
}


def _coerce(df):
    for col, dtype in RESULT_SCHEMA.items():
        if col not in df.columns:
            df[col] = pd.Series([pd.NA] * len(df))
        df[col] = df[col].astype(dtype)
    return df[list(RESULT_SCHEMA.keys())]


def run_comparison(state, feature_sets, classifiers, win_sec, overlap_pct,
                   subject="unknown", user_id="app", n_splits=5, progress=None):
    """
    window/overlap and CV fixed; only feature x classifier vary.
    """
    run_id = str(uuid.uuid4())[:8]
    now = dt.datetime.utcnow()
    total = max(1, len(feature_sets) * len(classifiers))
    rows, done = [], 0

    for feats in feature_sets:
        for clf in classifiers:
            if progress is not None:
                progress(done / total, desc=f"{'+'.join(feats)} · {clf}")
            try:
                r = evaluate(state["epochs_data"], state["labels"],
                             state["sfreq"], win_sec=win_sec,
                             overlap_pct=overlap_pct, selected_extractors=feats,
                             classifier=clf, n_splits=n_splits)
            except Exception as e:
                print(f"[skip] {feats} {clf}: {e}")
                done += 1
                continue
            rows.append({
                "run_id": run_id, "timestamp_utc": now, "user_id": user_id,
                "subject": subject, "feature_method": "+".join(feats),
                "classifier": clf, "window_sec": float(win_sec),
                "overlap_pct": float(overlap_pct), "cv_folds": int(n_splits),
                "n_windows": r["n_windows"], "n_trials": r["n_trials"],
                "mean_accuracy": r["mean_accuracy"],
                "std_accuracy": r["std_accuracy"], "rank": 0,
            })
            done += 1

    if not rows:
        return _coerce(pd.DataFrame()), None

    df = pd.DataFrame(rows).sort_values("mean_accuracy",
                                        ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df = _coerce(df)

    blob_path = None
    if BLOB_ENABLED:
        ts = now.strftime("%Y%m%dT%H%M%S")
        blob_path = upload_dataframe_parquet(
            df, RESULTS, f"results_{subject}_{ts}_{run_id}.parquet")
    return df, blob_path
