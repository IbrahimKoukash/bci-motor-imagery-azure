"""

Decoding algorithms - extends the decoding backbone.

Import after decoding.py so it can extend the CLASSIFIERS dict and reuse
make_windows / build_pipeline. Registers five classifiers (LDA, SVM, MLP,
Random Forest, KNN) and adds evaluate_full(), which returns predictions so a
confusion matrix can be built. Leak-safe: GroupKFold on trial id, predictions
collected out-of-fold via cross_val_predict.

"""
import numpy as np

from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# decoding.py
from decoding import make_windows, build_pipeline, CLASSIFIERS

CLASS_NAMES = {1: "left_hand", 2: "right_hand", 3: "feet", 4: "tongue"}


# Register all five classifiers 
CLASSIFIERS["lda"] = lambda: LinearDiscriminantAnalysis()
CLASSIFIERS["svm"] = lambda: SVC(kernel="rbf", C=1.0, gamma="scale")
CLASSIFIERS["mlp"] = lambda: MLPClassifier(
    hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
    alpha=1e-3, batch_size=32, learning_rate_init=1e-3, max_iter=300,
    early_stopping=True, n_iter_no_change=20, random_state=42)
CLASSIFIERS["rf"] = lambda: RandomForestClassifier(
    n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42)
CLASSIFIERS["knn"] = lambda: KNeighborsClassifier(
    n_neighbors=7, weights="distance")


# Evaluate with predictions -> accuracy + confusion matrix 
def evaluate_full(epochs_data, labels, sfreq, win_sec, overlap_pct,
                  selected_extractors, classifier="lda", n_splits=5):
    X, y, groups = make_windows(epochs_data, labels, sfreq, win_sec, overlap_pct)
    pipe = build_pipeline(selected_extractors, classifier, sfreq=sfreq)
    cv = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))

    # every window predicted once, on the fold where it was held out
    y_pred = cross_val_predict(pipe, X, y, groups=groups, cv=cv)

    classes = np.unique(y)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred, labels=classes)
    report = classification_report(
        y, y_pred, labels=classes,
        target_names=[CLASS_NAMES.get(c, str(c)) for c in classes],
        output_dict=True, zero_division=0)

    return {"accuracy": float(acc),
            "confusion_matrix": cm,
            "classes": classes,
            "y_true": y, "y_pred": y_pred,
            "report": report,
            "n_windows": int(len(y)),
            "n_trials": int(len(np.unique(groups)))}
