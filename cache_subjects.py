"""
Pre-cache preprocessed epochs for the sweep tab.

Run ONCE (offline). It preprocesses each subject (the slow part -
filter + ICA + epoch + ERD crop) and saves the resulting epochs array to a
small .npz file. The app's sweep tab then loads these instantly and only runs
the fast decoding part (features + CV), so reviewers get quick cross-subject
results without waiting on ICA.

    python cache_subjects.py --data-dir "C:\\path\\to\\BCICIV-2a-mat"

Output: cache/<subject>.npz  (one per subject), each holding
epochs_data, labels, sfreq. Bundle the cache/ folder with the app so the
sweep tab can read it (no Blob permissions needed).

"""
import os
import argparse
import numpy as np

from pipeline import preprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder with A0*T.mat")
    ap.add_argument("--out-dir", default="./cache")
    ap.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    subjects = [int(s) for s in args.subjects.split(",")]
    done = []

    for s in subjects:
        subj = f"A0{s}T"
        path = os.path.join(args.data_dir, f"{subj}.mat")
        if not os.path.exists(path):
            print(f"[{subj}] not found, skipping ({path})")
            continue
        print(f"[{subj}] preprocessing (filter + ICA + epoch + crop)…")
        try:
            res = preprocess(path, do_ica=True)
        except Exception as e:
            print(f"[{subj}] FAILED: {e}")
            continue

        out = os.path.join(args.out_dir, f"{subj}.npz")
        np.savez_compressed(
            out,
            epochs_data=res["epochs_data"].astype(np.float32),
            labels=res["labels"].astype(np.int16),
            sfreq=np.float64(res["sfreq"]),
            n_eog_removed=np.int16(res["n_eog_removed"]),
        )
        done.append(subj)
        d = res["epochs_data"]
        print(f"[{subj}] cached {d.shape} -> {out} "
              f"({os.path.getsize(out)/1e6:.1f} MB)")

    print(f"\nDone. {len(done)} subjects cached in {args.out_dir}: {done}")
    print("Bundle the cache/ folder with the app (it's picked up by the "
          "Dockerfile's COPY . .) so the sweep tab can load it.")


if __name__ == "__main__":
    main()
