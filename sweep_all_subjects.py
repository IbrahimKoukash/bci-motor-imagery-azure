"""
Multi-subject sweep — runs the full grid across all 9 subjects and writes
one Fabric-ready Parquet file per subject to the Blob 'results' container.

Uses the SAME modules as the app (pipeline, decoding, decoding_algorithms,
storage), so results are identical to what the app produces. Run this once your
single-subject accuracy is confirmed good; the output is the data the Fabric
Lakehouse + Power BI report read.

    python sweep_all_subjects.py --data-dir "C:\\path\\to\\BCICIV-2a-mat"

Set the same env vars as the app so results land in the right place:
    $env:BLOB_ACCOUNT_URL="https://kyudaideeptechstoragev2.blob.core.windows.net"
    $env:BLOB_CONTAINER_PREFIX="ibrahim-"
Omit BLOB_ACCOUNT_URL (or set LOCAL_ONLY=1) to write Parquet locally instead.

"""
import os
import argparse
import datetime as dt
import uuid
import traceback

import numpy as np
import pandas as pd

import decoding  # (evaluate, EXTRACTORS, CLASSIFIERS)
import decoding_algorithms # registers rf/knn/mlp
from decoding import evaluate
from pipeline import preprocess
from storage import BLOB_ENABLED, RESULTS, upload_dataframe_parquet

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


# the grid 
FEATURE_SETS = [
    ["freq_bandpower"], ["spatial_csp"],
]
CLASSIFIERS = ["lda", "svm"]
WINDOW_GRID = [(2.0, 0)] # (win_sec, overlap_pct)


def sweep_subject(X, y, sfreq, subject, n_splits=5, user_id="sweep"):
    run_id = str(uuid.uuid4())[:8]
    now = dt.datetime.utcnow()
    rows = []
    for win_sec, ovl in WINDOW_GRID:
        for feats in FEATURE_SETS:
            for clf in CLASSIFIERS:
                try:
                    r = evaluate(X, y, sfreq, win_sec=win_sec, overlap_pct=ovl,
                                 selected_extractors=feats, classifier=clf,
                                 n_splits=n_splits)
                except Exception as e:
                    print(f"    [skip] {subject} {feats} {clf} "
                          f"w{win_sec}/o{ovl}: {e}")
                    continue
                rows.append({
                    "run_id": run_id, "timestamp_utc": now, "user_id": user_id,
                    "subject": subject, "feature_method": "+".join(feats),
                    "classifier": clf, "window_sec": float(win_sec),
                    "overlap_pct": float(ovl), "cv_folds": int(n_splits),
                    "n_windows": r["n_windows"], "n_trials": r["n_trials"],
                    "mean_accuracy": r["mean_accuracy"],
                    "std_accuracy": r["std_accuracy"], "rank": 0})
                print(f"    {subject} {'+'.join(feats):28s} {clf:4s} "
                      f"w{win_sec}/o{ovl:>2}  acc={r['mean_accuracy']:.3f}")
    df = pd.DataFrame(rows)
    if df.empty:
        return _coerce(df)
    df = df.sort_values("mean_accuracy", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return _coerce(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder with A0*T.mat")
    ap.add_argument("--out-dir", default="./results_store",
                    help="local fallback when Blob is off")
    ap.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9")
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    subjects = [int(s) for s in args.subjects.split(",")]
    written = []

    for s in subjects:
        subj = f"A0{s}T"
        path = os.path.join(args.data_dir, f"{subj}.mat")
        if not os.path.exists(path):
            print(f"[{subj}] not found, skipping ({path})")
            continue
        print(f"\n[{subj}] preprocessing…")
        try:
            res = preprocess(path, do_ica=True)
            X, y, sfreq = res["epochs_data"], res["labels"], res["sfreq"]
            print(f"[{subj}] {X.shape} · classes "
                  f"{dict(zip(*np.unique(y, return_counts=True)))}")
        except Exception as e:
            print(f"[{subj}] preprocessing FAILED: {e}")
            traceback.print_exc()
            continue

        print(f"[{subj}] sweeping…")
        df = sweep_subject(X, y, sfreq, subj, n_splits=args.n_splits)
        if df.empty:
            print(f"[{subj}] no results")
            continue

        ts = pd.Timestamp(df["timestamp_utc"].iloc[0]).strftime("%Y%m%dT%H%M%S")
        fname = f"results_{subj}_{ts}_{df['run_id'].iloc[0]}.parquet"

        if BLOB_ENABLED:
            path_out = upload_dataframe_parquet(df, RESULTS, fname)
        else:
            path_out = os.path.join(args.out_dir, fname)
            df.to_parquet(path_out, engine="pyarrow", index=False)

        written.append(path_out)
        b = df.iloc[0]
        print(f"[{subj}] wrote {len(df)} rows -> {path_out}")
        print(f"[{subj}] BEST {b['feature_method']}+{b['classifier']} "
              f"(w{b['window_sec']}/o{b['overlap_pct']}) = {b['mean_accuracy']:.3f}")

    print(f"\nDone. {len(written)} subject files written.")
    if BLOB_ENABLED:
        print(f"They are in the Blob '{RESULTS}' container — point the Fabric "
              f"Lakehouse there.")
    else:
        print(f"Local files in {args.out_dir}. Upload them to the Blob "
              f"'{RESULTS}' container for Fabric.")


if __name__ == "__main__":
    main()
