"""

Param-aware pipeline building.

Wraps the existing extractors/classifiers so user-chosen parameters (from the
Advanced tab) flow into them. Falls back to defaults for anything not tuned.
Keeps the leak-safe structure: params are baked into the estimators, everything
still lives inside one sklearn Pipeline fitted per fold.

"""
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_score

from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from decoding import (make_windows, EXTRACTORS,
                      VarianceFeatures, BandPowerFeatures, CSPFeatures)
try:
    from decoding import (HjorthFeatures, SpectralEntropyFeatures,
                          STFTBandEnergy, WaveletPower, LaplacianBandPower)
except Exception:
    HjorthFeatures = SpectralEntropyFeatures = None
    STFTBandEnergy = WaveletPower = LaplacianBandPower = None

from params import default_params


# Build one extractor instance with user params 
def _make_extractor(name, sfreq, ep):
    """
    
    ep = this extractor's param dict (may be empty).
    
    """
    if name == "freq_bandpower":
        bands = ((ep.get("mu_lo", 8), ep.get("mu_hi", 12)),
                 (ep.get("beta_lo", 12), ep.get("beta_hi", 30)))
        return BandPowerFeatures(sfreq=sfreq, bands=bands)
    if name == "spatial_csp":
        return CSPFeatures(n_components=ep.get("n_components", 4))
    if name == "tf_stft" and STFTBandEnergy is not None:
        return STFTBandEnergy(sfreq=sfreq, nperseg=ep.get("nperseg", 64))
    if name == "spatial_laplacian" and LaplacianBandPower is not None:
        return LaplacianBandPower(sfreq=sfreq,
                                  band=(ep.get("band_lo", 8), ep.get("band_hi", 30)))
    # no-param extractors: construct via the registry 
    ctor, _ = EXTRACTORS[name]
    try:
        return ctor(sfreq=sfreq)
    except TypeError:
        return ctor()


# Build one classifier with user params 
def _make_classifier(name, cp):
    if name == "lda":
        return LinearDiscriminantAnalysis()
    if name == "svm":
        return SVC(kernel=cp.get("kernel", "rbf"), C=cp.get("C", 1.0),
                   gamma="scale")
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=cp.get("n_neighbors", 7),
                                    weights="distance")
    if name == "rf":
        md = cp.get("max_depth", 0)
        return RandomForestClassifier(
            n_estimators=cp.get("n_estimators", 300),
            max_depth=(None if md in (0, None) else md),
            min_samples_leaf=2, n_jobs=-1, random_state=42)
    if name == "mlp":
        h = cp.get("hidden", 64)
        return MLPClassifier(hidden_layer_sizes=(h, max(8, h // 2)),
                             activation="relu", solver="adam",
                             alpha=cp.get("alpha", 1e-3), batch_size=32,
                             learning_rate_init=1e-3, max_iter=300,
                             early_stopping=True, n_iter_no_change=20,
                             random_state=42)
    raise KeyError(name)


def build_pipeline_p(selected_extractors, classifier, sfreq, params):
    """
    Param-aware build. params = {'extractors':{}, 'classifiers':{}}.
    
    """
    if not selected_extractors:
        raise ValueError("select at least one feature extractor")
    ex_params = params.get("extractors", {})
    transformers = [(name, _make_extractor(name, sfreq, ex_params.get(name, {})))
                    for name in selected_extractors]
    clf = _make_classifier(classifier,
                           params.get("classifiers", {}).get(classifier, {}))
    return Pipeline([("features", FeatureUnion(transformers)),
                     ("scaler", StandardScaler()),
                     ("clf", clf)])


def evaluate_p(epochs_data, labels, sfreq, win_sec, overlap_pct,
               selected_extractors, classifier="lda", n_splits=5, params=None):
    if params is None:
        params = default_params()
    X, y, groups = make_windows(epochs_data, labels, sfreq, win_sec, overlap_pct)
    pipe = build_pipeline_p(selected_extractors, classifier, sfreq, params)
    cv = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    scores = cross_val_score(pipe, X, y, groups=groups, cv=cv, scoring="accuracy")
    return {"mean_accuracy": float(scores.mean()),
            "std_accuracy": float(scores.std()),
            "fold_scores": scores.tolist(),
            "n_windows": int(len(y)),
            "n_trials": int(len(np.unique(groups)))}
