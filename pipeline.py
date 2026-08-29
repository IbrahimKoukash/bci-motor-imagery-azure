"""

BCIC IV 2a preprocessing: .mat -> cleaned, ERD-cropped epochs.

"""
import numpy as np
import pandas as pd
import scipy.io as sio

CH_EEG = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
          'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
          'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
          'P1', 'Pz', 'P2', 'POz']
CH_EOG = ['EOG-left', 'EOG-central', 'EOG-right']
CH_ALL = CH_EEG + CH_EOG
CLASS_MAP = {1: 'left_hand', 2: 'right_hand', 3: 'feet', 4: 'tongue'}


def load_bci2a(path):
    """
    
    (raws list, trials DataFrame).
    Runs without MI trials give raws[i]=None.
    
    """
    import mne
    mne.set_log_level("ERROR")

    mat = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    runs = mat['data']

    raws, rows = [], []
    for run_idx, run in enumerate(runs):
        X = np.asarray(run.X, dtype=np.float64)
        if X.size == 0 or X.ndim != 2 or X.shape[1] < 25:
            raws.append(None)
            continue
        trial = np.atleast_1d(np.asarray(run.trial)).astype(int)
        y = np.atleast_1d(np.asarray(run.y)).astype(int)
        fs = int(run.fs)

        info = mne.create_info(ch_names=CH_ALL, sfreq=fs,
                               ch_types=['eeg'] * 22 + ['eog'] * 3)
        raw = mne.io.RawArray(X.T * 1e-6, info, verbose=False)
        raw.set_montage(mne.channels.make_standard_montage('standard_1005'),
                        on_missing='ignore', verbose=False)
        raws.append(raw)

        for start, lab in zip(trial, y):
            if int(lab) in CLASS_MAP:
                rows.append({'run': run_idx, 'start_sample': int(start),
                             'label': int(lab), 'class': CLASS_MAP[int(lab)]})

    trials_df = pd.DataFrame(rows)
    if trials_df.empty:
        raise RuntimeError("No motor-imagery trials found in this file.")
    return raws, trials_df


def preprocess(path, l_freq=1.0, h_freq=40.0, n_ica=20, cue_shift=2.0,
               erd_tmin=0.5, erd_tmax=2.5, do_ica=True, progress=None):
    """
    
    Full pipeline -> dict with epochs_data, labels, sfreq, and diagnostics.
    
    """
    import mne
    from mne.preprocessing import ICA
    mne.set_log_level("ERROR")

    def _p(frac, msg):
        if progress is not None:
            progress(frac, desc=msg)

    _p(0.05, "loading .mat")
    raws, trials_df = load_bci2a(path)

    # MI runs
    mi_runs = sorted(trials_df['run'].unique())
    _p(0.20, f"MI runs: {mi_runs}")

    raw = mne.concatenate_raws([raws[i].copy() for i in mi_runs])

    _p(0.30, "FIR band-pass + average reference")
    raw.filter(l_freq=l_freq, h_freq=h_freq, method='fir', fir_design='firwin',
               phase='zero', fir_window='hamming', verbose=False)
    raw.set_eeg_reference('average', projection=False, verbose=False)

    eog_idx, eog_scores = [], []
    if do_ica:
        _p(0.45, "fitting ICA")
        ica = ICA(n_components=n_ica, method='fastica', random_state=42,
                  max_iter='auto', verbose=False)
        ica.fit(raw)
        try:
            eog_idx, eog_scores = ica.find_bads_eog(raw, verbose=False)
        except Exception:
            eog_idx, eog_scores = [], []
        ica.exclude = list(eog_idx)
    else:
        ica = None

    _p(0.75, "epoching (cue-locked)")
    sfreq = raw.info['sfreq']
    run_lengths = [raws[i].n_times for i in mi_runs]
    offsets = np.cumsum([0] + run_lengths[:-1])
    run_to_offset = dict(zip(mi_runs, offsets))

    ev = trials_df[trials_df['run'].isin(mi_runs)].copy()
    ev['abs'] = ev.apply(lambda r: r['start_sample'] + run_to_offset[r['run']],
                         axis=1) + int(round(cue_shift * sfreq))
    events = np.c_[ev['abs'].values.astype(int),
                   np.zeros(len(ev), int),
                   ev['label'].values.astype(int)]

    epochs = mne.Epochs(raw, events,
                        event_id={'left_hand': 1, 'right_hand': 2,
                                  'feet': 3, 'tongue': 4},
                        tmin=-1.0, tmax=4.0, baseline=None,
                        picks=['eeg', 'eog'], preload=True, verbose=False)

    _p(0.88, "applying ICA + baseline")
    epochs_clean = ica.apply(epochs.copy(), verbose=False) if do_ica else epochs
    epochs_clean.apply_baseline((-0.5, 0.0), verbose=False)

    _p(0.95, f"cropping to ERD window {erd_tmin}-{erd_tmax}s")
    ep = epochs_clean.copy().pick('eeg').crop(tmin=erd_tmin, tmax=erd_tmax)

    counts = trials_df['class'].value_counts().to_dict()
    return {
        "epochs_data": ep.get_data(),
        "labels": ep.events[:, 2],
        "sfreq": float(sfreq),
        "epochs_clean": epochs_clean,
        "raw_filt": raw,
        "mi_runs": mi_runs,
        "n_eog_removed": len(eog_idx),
        "eog_idx": list(eog_idx),
        "eog_scores": eog_scores,   
        "ica": ica,                   
        "class_counts": counts,
        "erd_window": (erd_tmin, erd_tmax),
    }
