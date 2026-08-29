"""

Extra visualizations 

Four functions, each returns a matplotlib Figure for st.pyplot:
  1. plot_ica_eog() - which components were removed as eye artifacts,
                              their correlation with EOG, and before/after.
  2. plot_erd_topomaps() - per-class mu/beta power scalp maps (where in the
                              brain each movement shows up).
  3. plot_confusion() - confusion matrix for one classifier.
  4. plot_montage() -  Electrodes placement on the scalp.

"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Theme 
GROUND = "#0E1621"; PANEL = "#16212E"; INK = "#E6EEF5"; MUTED = "#8AA0B4"
SIGNAL = "#31E0C8"; AMBER = "#E7B45A"; LINE = "#26374A"
CLASS_NAMES = {1: "left_hand", 2: "right_hand", 3: "feet", 4: "tongue"}
CLASS_COLORS = {1: "#31E0C8", 2: "#E7B45A", 3: "#7FB2FF", 4: "#F58F8F"}


def _dark(fig, axes):
    fig.patch.set_facecolor(GROUND)
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor(PANEL)
        for s in ax.spines.values():
            s.set_color(LINE)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(INK)


# ICA / EOG transparency - what was removed and why
def plot_ica_eog(ica, epochs_clean, eog_idx, eog_scores):
    """
    Show, per flagged component, its |correlation| with the EOG channels,
    plus the component topographies MNE flagged. If nothing was flagged,
    returns a small 'no eye artifacts detected' figure.

    """
    scores = np.atleast_2d(eog_scores) # (n_eog_ch, n_components)
    n_comp = scores.shape[1]

    fig = plt.figure(figsize=(11, 4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1])

    # correlation bar chart across ALL components, flagged ones highlighted
    ax = fig.add_subplot(gs[0, 0])
    max_corr = np.max(np.abs(scores), axis=0) # strongest EOG corr per comp
    colors = [AMBER if i in eog_idx else LINE for i in range(n_comp)]
    ax.bar(range(n_comp), max_corr, color=colors)
    ax.axhline(0, color=MUTED, lw=.5)
    ax.set_xlabel("ICA component"); ax.set_ylabel("|corr with EOG|")
    ax.set_title("Eye-artifact detection (amber = removed)")
    for i in eog_idx:
        ax.text(i, max_corr[i] + .02, f"{max_corr[i]*100:.0f}%",
                ha="center", color=AMBER, fontsize=8)

    # Short textual summary panel
    ax2 = fig.add_subplot(gs[0, 1]); ax2.axis("off")
    lines = [f"Components removed: {len(eog_idx)}"]
    for i in eog_idx:
        lines.append(f"  IC{i:02d}   {max_corr[i]*100:.0f}% correlated with EOG")
    if not eog_idx:
        lines = ["No eye-artifact components detected.",
                 "(EOG correlation below threshold.)"]
    ax2.text(0.0, 0.95, "\n".join(lines), va="top", family="monospace",
             color=INK, fontsize=10, transform=ax2.transAxes)

    _dark(fig, [ax, ax2])
    fig.tight_layout()
    return fig


def plot_ica_topographies(ica, eog_idx):
    """
    Component topographies for the removed components.
    Returns None if nothing was flagged (caller can skip).

    """
    if not eog_idx:
        return None
    fig = ica.plot_components(picks=eog_idx, show=False)
    try:
        fig.patch.set_facecolor(GROUND)
    except Exception:
        pass
    return fig


# Per-class ERD topomaps - where each movement shows up
def plot_erd_topomaps(epochs_clean, band=(8, 30)):
    """
    One scalp map per class: mean band power over the trial, per channel.
    stand out for the hand classes.

    """
    import mne
    classes = sorted(epochs_clean.event_id.values())
    names = [k for k, v in sorted(epochs_clean.event_id.items(),
                                  key=lambda kv: kv[1])]
    fig, axes = plt.subplots(1, len(classes), figsize=(3.1 * len(classes), 3.4))
    axes = np.atleast_1d(axes)

    for ax, cls, nm in zip(axes, classes, names):
        # pick EEG only so data (n_ch) matches info (n_ch) - drops the 3 EOG
        ep = (epochs_clean[nm].copy()
              .pick("eeg")                   
              .filter(band[0], band[1], verbose=False))
        data = ep.get_data() # (trials, 22, time)
        power = (data ** 2).mean(axis=(0, 2)) # (22,)
        power_db = 10 * np.log10(power / power.mean() + 1e-12)
        mne.viz.plot_topomap(power_db, ep.info, axes=ax, show=False,
                             cmap="RdBu_r", contours=4)
        ax.set_title(nm, color=INK, fontsize=11)

    fig.suptitle(f"Band-power topography {band[0]}-{band[1]} Hz  (per class)",
                 color=INK)
    fig.patch.set_facecolor(GROUND)
    fig.tight_layout()
    return fig



# Confusion matrix (row-normalised)
def plot_confusion(cm, classes, title="", normalize=True):
    names = [CLASS_NAMES.get(c, str(c)) for c in classes]
    cm = np.asarray(cm, float)
    if normalize:
        disp = cm / (cm.sum(axis=1, keepdims=True) + 1e-12); vmax = 1.0
        fmt = "{:.2f}"
    else:
        disp, vmax, fmt = cm, cm.max(), "{:.0f}"

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(disp, cmap="viridis", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45,
                                                         ha="right", fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title or "Confusion matrix")
    for i in range(disp.shape[0]):
        for j in range(disp.shape[1]):
            ax.text(j, i, fmt.format(disp[i, j]), ha="center", va="center",
                    color="white" if disp[i, j] < vmax * .6 else "black",
                    fontsize=9)
    _dark(fig, ax)
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.tight_layout()
    return fig

# Electrode montage - where the 22 channels sit on the head
def plot_montage(info):
    """
    2D sensor layout from the data's own montage.
    
    """
    import mne
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    fig.patch.set_facecolor(GROUND)
    ax.set_facecolor(GROUND)
    # EEG only — EOG channels have no scalp position
    picks = mne.pick_types(info, eeg=True, eog=False)
    info_eeg = mne.pick_info(info, picks)
    mne.viz.plot_sensors(info_eeg, kind="topomap", show_names=True, axes=ax,
                         show=False, pointsize=28, linewidth=0)
    ax.set_title("22 EEG electrode positions", color=INK, fontsize=11)
    # recolor the matplotlib text/markers to theme
    for txt in ax.texts:
        txt.set_color(INK); txt.set_fontsize(6.5)
    fig.tight_layout()
    return fig