"""

Curated parameter registry for extractors and classifiers.

Each entry lists only the key parameters worth tuning, with a
UI spec (type, range, default) so the Advanced-Settings tab can render controls
generically. The app collects these into a dict; build_pipeline / the classifier
factory read them. Parameters are set ONCE and fixed across the comparison grid.

PARAM_SPECS[name] = list of (key, label, kind, default, meta)
  kind: "slider_int" | "slider_float" | "select"
  meta: (min, max, step) for sliders, or [options] for select

"""

# feature-extractor parameters 
EXTRACTOR_PARAMS = {
    "freq_bandpower": [
        ("mu_lo",   "Mu band low (Hz)",   "slider_int", 8,  (4, 14, 1)),
        ("mu_hi",   "Mu band high (Hz)",  "slider_int", 12, (8, 16, 1)),
        ("beta_lo", "Beta band low (Hz)", "slider_int", 12, (12, 20, 1)),
        ("beta_hi", "Beta band high (Hz)","slider_int", 30, (20, 40, 1)),
    ],
    "spatial_csp": [
        ("n_components", "CSP components", "slider_int", 4, (2, 8, 1)),
    ],
    "tf_stft": [
        ("nperseg", "STFT window (samples)", "slider_int", 64, (32, 128, 16)),
    ],
    "spatial_laplacian": [
        ("band_lo", "Band low (Hz)",  "slider_int", 8,  (4, 14, 1)),
        ("band_hi", "Band high (Hz)", "slider_int", 30, (20, 40, 1)),
    ],
    # time_variance, time_hjorth, freq_entropy, tf_wavelet: no key params to tune
}

# classifier parameters 
CLASSIFIER_PARAMS = {
    "svm": [
        ("C",      "C (regularization)", "select", 1.0, [0.1, 1.0, 10.0]),
        ("kernel", "Kernel", "select", "rbf", ["rbf", "linear"]),
    ],
    "knn": [
        ("n_neighbors", "Neighbors (k)", "slider_int", 7, (3, 15, 2)),
    ],
    "rf": [
        ("n_estimators", "Trees", "slider_int", 300, (100, 500, 100)),
        ("max_depth",    "Max depth (0=none)", "slider_int", 0, (0, 20, 2)),
    ],
    "mlp": [
        ("hidden", "Hidden units (layer 1)", "slider_int", 64, (16, 128, 16)),
        ("alpha",  "L2 penalty", "select", 1e-3, [1e-4, 1e-3, 1e-2]),
    ],
    # lda: no key params
}


def default_params():
    """
    
    Return the full default param dict, so the app works untouched.
    
    """
    out = {"extractors": {}, "classifiers": {}}
    for name, specs in EXTRACTOR_PARAMS.items():
        out["extractors"][name] = {k: d for (k, _l, _t, d, _m) in specs}
    for name, specs in CLASSIFIER_PARAMS.items():
        out["classifiers"][name] = {k: d for (k, _l, _t, d, _m) in specs}
    return out
