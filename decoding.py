"""

Leak-safe decoding backbone for the BCIC IV 2a pipeline.

Design principles:
  1. Windowing produces (window, features) samples but every window carries the
     trial id it came from. Cross-validation splits on trial id (GroupKFold), so
     windows from one trial never straddle train/test.
  2. Every step that learns from data (scaler, CSP, classifier) lives inside an
     sklearn Pipeline, so it is re-fit on the training fold only. No feature is
     ever computed on the full dataset before splitting.
  3. Feature extractors are transformers that operate on epoch arrays of shape
     (n_windows, n_channels, n_times). User-selectable extractors are combined
     with FeatureUnion. Supervised extractors (CSP) sit in the same pipeline so
     they too are fit per fold.

The Azure UI plugs in by choosing: window length, overlap %, which extractors,
and which classifier. Everything below is what those choices assemble.

"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# MNE's CSP is preferred
try:
    from mne.decoding import CSP as _MNE_CSP
    _HAS_MNE = True
except Exception:
    _HAS_MNE = False


# Windowing  ->  produces X (n_windows, n_ch, n_win_samples), y, groups
def make_windows(epochs_data, labels, sfreq, win_sec, overlap_pct):
    """
    Slice each trial into overlapping windows.

    epochs_data: (n_trials, n_channels, n_times) for example, epochs_clean.get_data()
    labels: (n_trials,) integer class per trial
    sfreq: sampling rate (250 for 2a)
    win_sec: window length in seconds (user-chosen)
    overlap_pct: overlap in percent 0<100 (user-chosen)

    Returns X_win (n_windows, n_ch, win_samples), y_win, groups (trial id).
    'groups' is the anti-leak key: it ties every window back to its trial.

    """
    n_trials, n_ch, n_times = epochs_data.shape
    win = int(round(win_sec * sfreq))
    if win <= 0 or win > n_times:
        raise ValueError(f"window ({win} samples) must be in 1..{n_times}")
    step = max(1, int(round(win * (1.0 - overlap_pct / 100.0))))

    Xs, ys, groups = [], [], []
    for t in range(n_trials):
        starts = range(0, n_times - win + 1, step)
        for s in starts:
            Xs.append(epochs_data[t, :, s:s + win])
            ys.append(labels[t])
            groups.append(t)  # trial id travels with the window
    return np.asarray(Xs), np.asarray(ys), np.asarray(groups)


# Feature extractors  (transformers on (n_win, n_ch, n_times))
# Two per domain, all user-selectable, all combinable via FeatureUnion.
# Unsupervised ones (everything except CSP) are stateless-ish transforms.
class _BaseEpochTransformer(BaseEstimator, TransformerMixin):
    """
    Fit is a no-op for unsupervised extractors; keeps the sklearn contract.
    """
    def fit(self, X, y=None):
        return self


# Time Domain 
class VarianceFeatures(_BaseEpochTransformer):
    """
    Log-variance per channel. Cheap, strong baseline for band-limited data.
    """
    def transform(self, X):
        return np.log(np.var(X, axis=2) + 1e-12) # (n_win, n_ch)


class HjorthFeatures(_BaseEpochTransformer):
    """
    Hjorth activity, mobility, complexity per channel.
    """
    def transform(self, X):
        d1 = np.diff(X, axis=2)
        d2 = np.diff(d1, axis=2)
        var0 = np.var(X, axis=2) + 1e-12
        var1 = np.var(d1, axis=2) + 1e-12
        var2 = np.var(d2, axis=2) + 1e-12
        activity = var0
        mobility = np.sqrt(var1 / var0)
        complexity = np.sqrt(var2 / var1) / (mobility + 1e-12)
        return np.concatenate([activity, mobility, complexity], axis=1)


# Frequency Domain 
class BandPowerFeatures(_BaseEpochTransformer):
    """
    Average PSD power in named bands per channel (Welch via rFFT).
    """
    def __init__(self, sfreq=250.0, bands=((8, 12), (12, 30))):
        self.sfreq = sfreq
        self.bands = bands

    def transform(self, X):
        n_win, n_ch, n_t = X.shape
        freqs = np.fft.rfftfreq(n_t, d=1.0 / self.sfreq)
        psd = (np.abs(np.fft.rfft(X, axis=2)) ** 2) / n_t
        feats = []
        for lo, hi in self.bands:
            m = (freqs >= lo) & (freqs < hi)
            feats.append(psd[:, :, m].mean(axis=2))
        return np.concatenate(feats, axis=1) # (n_win, n_ch*n_bands)


class SpectralEntropyFeatures(_BaseEpochTransformer):
    """
    Shannon entropy of the normalized power spectrum per channel.
    """
    def __init__(self, sfreq=250.0):
        self.sfreq = sfreq

    def transform(self, X):
        n_win, n_ch, n_t = X.shape
        psd = np.abs(np.fft.rfft(X, axis=2)) ** 2
        p = psd / (psd.sum(axis=2, keepdims=True) + 1e-12)
        ent = -np.sum(p * np.log(p + 1e-12), axis=2)
        return ent # (n_win, n_ch)


# Time-Frequency Domain 
class STFTBandEnergy(_BaseEpochTransformer):
    """
    Energy in bands from a short-time FFT, averaged over time frames.
    """
    def __init__(self, sfreq=250.0, bands=((8, 12), (12, 30)), nperseg=64):
        self.sfreq = sfreq
        self.bands = bands
        self.nperseg = nperseg

    def transform(self, X):
        from scipy.signal import stft
        n_win, n_ch, n_t = X.shape
        nps = min(self.nperseg, n_t)
        feats = []
        f, _, Z = stft(X, fs=self.sfreq, nperseg=nps, axis=2)
        P = np.abs(Z) ** 2 # (n_win,n_ch,n_f,n_frames)
        for lo, hi in self.bands:
            m = (f >= lo) & (f < hi)
            feats.append(P[:, :, m, :].mean(axis=(2, 3)))
        return np.concatenate(feats, axis=1)


class WaveletPower(_BaseEpochTransformer):
    """
    Morlet wavelet power per band via scipy.signal.cwt (Ricker approx).
    """
    def __init__(self, sfreq=250.0, widths=(4, 8, 16)):
        self.sfreq = sfreq
        self.widths = widths

    def transform(self, X):
        from scipy.signal import cwt, ricker
        n_win, n_ch, n_t = X.shape
        out = np.empty((n_win, n_ch * len(self.widths)))
        for i in range(n_win):
            cols = []
            for c in range(n_ch):
                coef = cwt(X[i, c], ricker, self.widths)
                cols.append(np.log(np.mean(coef ** 2, axis=1) + 1e-12))
            out[i] = np.concatenate(cols)
        return out


# Spatial Domain 
class CSPFeatures(BaseEstimator, TransformerMixin):
    """
    Common Spatial Patterns. SUPERVISED -> must be fit inside the CV fold.
    Uses MNE's CSP when available; otherwise a compact NumPy CSP (binary or
    one-vs-rest averaged). Kept as a normal pipeline step so refit-per-fold
    happens automatically.

    """
    def __init__(self, n_components=4):
        self.n_components = n_components

    def fit(self, X, y):
        if _HAS_MNE:
            self._csp = _MNE_CSP(n_components=self.n_components,
                                 reg='ledoit_wolf', log=True, norm_trace=False)
            self._csp.fit(X, y)
            return self
        #  NumPy fallback (binary or OVR) 
        classes = np.unique(y)
        def cov(a):
            a = a - a.mean(axis=1, keepdims=True)
            c = a @ a.T
            return c / np.trace(c)
        filters = []
        for cl in classes:                       # one-vs-rest
            Ci = np.mean([cov(X[i]) for i in np.where(y == cl)[0]], axis=0)
            Cr = np.mean([cov(X[i]) for i in np.where(y != cl)[0]], axis=0)
            evals, evecs = np.linalg.eig(np.linalg.pinv(Cr + Ci) @ Ci)
            order = np.argsort(evals)[::-1]
            filters.append(evecs[:, order[:self.n_components]].real.T)
        self._filters = np.concatenate(filters, axis=0)
        return self

    def transform(self, X):
        if _HAS_MNE:
            return self._csp.transform(X)
        feats = np.empty((X.shape[0], self._filters.shape[0]))
        for i in range(X.shape[0]):
            z = self._filters @ X[i]
            feats[i] = np.log(np.var(z, axis=1) + 1e-12)
        return feats


class LaplacianBandPower(_BaseEpochTransformer):
    """
    Small-Laplacian-style spatial sharpening (center minus neighbor mean) then
    band power. Unsupervised, so safe outside the fold, but kept in-pipeline
    for consistency. neighbor_idx maps each channel to its neighbor indices.

    """
    def __init__(self, sfreq=250.0, band=(8, 30), neighbor_idx=None):
        self.sfreq = sfreq
        self.band = band
        self.neighbor_idx = neighbor_idx

    def transform(self, X):
        n_win, n_ch, n_t = X.shape
        if self.neighbor_idx is None:
            Xl = X
        else:
            Xl = np.empty_like(X)
            for c in range(n_ch):
                nb = self.neighbor_idx.get(c, [])
                Xl[:, c, :] = X[:, c, :] - (X[:, nb, :].mean(axis=1) if nb else 0)
        freqs = np.fft.rfftfreq(n_t, d=1.0 / self.sfreq)
        psd = np.abs(np.fft.rfft(Xl, axis=2)) ** 2
        m = (freqs >= self.band[0]) & (freqs < self.band[1])
        return np.log(psd[:, :, m].mean(axis=2) + 1e-12)


# registry the UI reads from: name -> (constructor, is_supervised)
EXTRACTORS = {
    "time_variance": (VarianceFeatures, False),
    "time_hjorth": (HjorthFeatures, False),
    "freq_bandpower": (BandPowerFeatures, False),
    "freq_entropy": (SpectralEntropyFeatures, False),
    "tf_stft": (STFTBandEnergy, False),
    "tf_wavelet": (WaveletPower, False),
    "spatial_csp": (CSPFeatures, True), # supervised
    "spatial_laplacian": (LaplacianBandPower, False),
}

CLASSIFIERS = {
    "lda": lambda: LinearDiscriminantAnalysis(),
    "svm": lambda: SVC(kernel="rbf", C=1.0, gamma="scale"),
}


# Assemble THE Leak-safe Pipeline from user choices
def build_pipeline(selected_extractors, classifier="lda", sfreq=250.0,
                   csp_components=4):
    """
    selected_extractors: list of keys from EXTRACTORS (user's checkbox picks)
    classifier: key from CLASSIFIERS
    Returns an sklearn Pipeline: FeatureUnion(extractors) -> scaler -> clf.
    CSP, if selected, sits inside the union and is thus re-fit per fold.

    """
    if not selected_extractors:
        raise ValueError("select at least one feature extractor")

    transformers = []
    for name in selected_extractors:
        ctor, _ = EXTRACTORS[name]
        # pass sfreq to the ones that need it
        try:
            inst = ctor(sfreq=sfreq)
        except TypeError:
            inst = ctor()
        if name == "spatial_csp":
            inst = CSPFeatures(n_components=csp_components)
        transformers.append((name, inst))

    union = FeatureUnion(transformers)
    return Pipeline([
        ("features", union),
        ("scaler", StandardScaler()), # fit on train fold only
        ("clf", CLASSIFIERS[classifier]()),
    ])


# Evaluate with GroupKFold on Trial ids
def evaluate(epochs_data, labels, sfreq, win_sec, overlap_pct,
             selected_extractors, classifier="lda", n_splits=5):
    X, y, groups = make_windows(epochs_data, labels, sfreq, win_sec, overlap_pct)
    pipe = build_pipeline(selected_extractors, classifier, sfreq=sfreq)
    cv = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    scores = cross_val_score(pipe, X, y, groups=groups, cv=cv, scoring="accuracy")
    return {
        "mean_accuracy": float(scores.mean()),
        "std_accuracy": float(scores.std()),
        "fold_scores": scores.tolist(),
        "n_windows": int(len(y)),
        "n_trials": int(len(np.unique(groups))),
    }


#  MLP registered as a classifier
from sklearn.neural_network import MLPClassifier
CLASSIFIERS["mlp"] = lambda: MLPClassifier(
    hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
    alpha=1e-3, batch_size=32, learning_rate_init=1e-3, max_iter=300,
    early_stopping=True, n_iter_no_change=20, random_state=42)
