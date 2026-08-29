"""
User-controlled cross-subject sweep, reading pre-cached epochs.

The heavy preprocessing was done once by cache_subjects.py. Here we only run
the FAST part (windowing + features + CV) for the subjects/features/classifiers
the reviewer picks, so it stays responsive. Returns a tidy DataFrame and
cross-subject plots for the sweep tab.

"""
import os
import glob
import uuid
import datetime as dt
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from decoding import evaluate # fast decode + CV

# theme (match the app)
GROUND = "#0E1621"; PANEL = "#16212E"; INK = "#E6EEF5"; MUTED = "#8AA0B4"
SIGNAL = "#31E0C8"; AMBER = "#E7B45A"; LINE = "#26374A"

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def available_subjects(cache_dir=CACHE_DIR):
    """List cached subject codes, e.g. ['A01T','A02T',...]."""
    if not os.path.isdir(cache_dir):
        return []
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(cache_dir, "*.npz")))


def _load(subject, cache_dir=CACHE_DIR):
    d = np.load(os.path.join(cache_dir, f"{subject}.npz"))
    return d["epochs_data"].astype(float), d["labels"].astype(int), float(d["sfreq"])


def run_sweep(subjects, feature_sets, classifiers, win_sec, overlap_pct,
              n_splits=5, params=None, progress=None):
    """
    subjects: list of cached subject codes
    feature_sets: list of lists (each inner list = one feature config)
    classifiers: list of classifier keys
    Bounded by the user's choices -> stays fast. One row per
    subject x feature_set x classifier.

    """
    # evaluate_p if params given, else plain evaluate
    if params is not None:
        from decoding_params import evaluate_p
        def _ev(X, y, sf, fs, clf):
            return evaluate_p(X, y, sf, win_sec, overlap_pct, fs, clf,
                              n_splits=n_splits, params=params)
    else:
        def _ev(X, y, sf, fs, clf):
            return evaluate(X, y, sf, win_sec, overlap_pct, fs, clf,
                            n_splits=n_splits)

    rows = []
    total = max(1, len(subjects) * len(feature_sets) * len(classifiers))
    done = 0
    for subj in subjects:
        X, y, sf = _load(subj)
        for feats in feature_sets:
            for clf in classifiers:
                if progress:
                    progress(done / total, desc=f"{subj} · {'+'.join(feats)} · {clf}")
                try:
                    r = _ev(X, y, sf, feats, clf)
                    rows.append({"subject": subj,
                                 "feature_method": "+".join(feats),
                                 "classifier": clf,
                                 "mean_accuracy": r["mean_accuracy"],
                                 "std_accuracy": r["std_accuracy"],
                                 "window_sec": win_sec,
                                 "overlap_pct": overlap_pct})
                except Exception as e:
                    print(f"[skip] {subj} {feats} {clf}: {e}")
                done += 1
    return pd.DataFrame(rows)

# Persist a sweep to Blob in the same schema the Fabric Lakehouse reads
def save_sweep_to_blob(df, user_id="reviewer", n_splits=5):
    """
    Coerce a run_sweep() frame to RESULT_SCHEMA and write one Parquet file to
    the results container. Returns the blob path, or None if Blob is off.
 
    Rank is computed Within each subject, so rank==1 means "the winning
    feature x classifier for that person" - which is what the Power BI
    win-count chart counts.
 
    """
    from comparison import RESULT_SCHEMA
    from storage import upload_dataframe_parquet, BLOB_ENABLED, RESULTS
 
    if not BLOB_ENABLED or df is None or len(df) == 0:
        return None
 
    run_id = str(uuid.uuid4())[:8]
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
 
    out = df.copy()
    out["run_id"] = run_id
    out["timestamp_utc"] = now
    out["user_id"] = user_id
    if "cv_folds" not in out.columns:
        out["cv_folds"] = n_splits
    out["rank"] = (out.groupby("subject")["mean_accuracy"]
                      .rank(ascending=False, method="first").astype("int64"))
 
    for col, dtype in RESULT_SCHEMA.items():
        if col not in out.columns:
            out[col] = pd.Series([pd.NA] * len(out))
        out[col] = out[col].astype(dtype)
    out = out[list(RESULT_SCHEMA.keys())]
 
    ts = now.strftime("%Y%m%dT%H%M%S")
    return upload_dataframe_parquet(
        out, RESULTS, f"results_sweep_{ts}_{run_id}.parquet")

# Theme
def _dark(fig, axes):
    fig.patch.set_facecolor(GROUND)
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor(PANEL)
        for s in ax.spines.values():
            s.set_color(LINE)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(INK)
        lg = ax.get_legend()
        if lg:
            lg.get_frame().set_facecolor(PANEL); lg.get_frame().set_edgecolor(LINE)
            for t in lg.get_texts():
                t.set_color(INK)

# Plotting
def plot_best_per_subject(df, chance=0.25):
    """
    Bar chart: best accuracy achieved for each subject.

    """
    if df is None or len(df) == 0:
        fig, ax = plt.subplots(figsize=(6, 2)); ax.axis("off")
        ax.text(.5, .5, "no results", ha="center", color=INK); _dark(fig, ax)
        return fig
    best = df.loc[df.groupby("subject")["mean_accuracy"].idxmax()]
    best = best.sort_values("mean_accuracy")
    fig, ax = plt.subplots(figsize=(8, max(3, .5 * len(best) + 1)))
    ax.barh(best["subject"], best["mean_accuracy"], color=SIGNAL)
    ax.axvline(chance, color=AMBER, ls="--", lw=1, label=f"chance {chance:.2f}")
    for i, (_, r) in enumerate(best.iterrows()):
        ax.text(r["mean_accuracy"] + .01, i,
                f"{r['feature_method']}·{r['classifier']}",
                va="center", fontsize=7, color=MUTED)
    ax.set_xlim(0, 1); ax.set_xlabel("best accuracy")
    ax.set_title("Best decoding accuracy per subject")
    ax.legend(fontsize=8)
    _dark(fig, ax)
    fig.tight_layout()
    return fig


def plot_subject_feature_heatmap(df):
    """
    Heatmap: subjects x feature_method (avg over classifiers).

    """
    if df is None or len(df) == 0:
        fig, ax = plt.subplots(figsize=(6, 2)); ax.axis("off"); return fig
    piv = df.pivot_table(index="subject", columns="feature_method",
                         values="mean_accuracy", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(1.4 * piv.shape[1] + 3, .5 * piv.shape[0] + 2))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto", vmin=0.25, vmax=1.0)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_title("Accuracy by subject x feature (avg over classifiers)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v < .7 else "black")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.patch.set_facecolor(GROUND); ax.title.set_color(INK)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    return fig
