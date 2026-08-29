"""
Dark theme - Every function returns a matplotlib figure for st.pyplot. 
App: plot_psd, plot_erd, plot_comparison

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

# Dark theme
def _dark(fig, axes):
    fig.patch.set_facecolor(GROUND)
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor(PANEL)
        for s in ax.spines.values():
            s.set_color(LINE)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(INK)
        if ax.get_legend():
            lg = ax.get_legend()
            lg.get_frame().set_facecolor(PANEL)
            lg.get_frame().set_edgecolor(LINE)
            for txt in lg.get_texts():
                txt.set_color(INK)

# Plotting
# Power Spectral Density (PSD)
def plot_psd(raw_filt):
    fig, ax = plt.subplots(figsize=(9, 3.3))
    try:
        raw_filt.compute_psd(fmax=60, picks="eeg").plot(
            axes=ax, show=False, color=SIGNAL)
    except TypeError:
        # older/newer MNE without color kwarg
        raw_filt.compute_psd(fmax=60, picks="eeg").plot(axes=ax, show=False)
    ax.set_title("Power spectrum after filtering (EEG)")
    _dark(fig, ax)
    fig.tight_layout()
    return fig

# Event Related Dysynchronization (ERD)
def plot_erd(epochs_clean, channels=("C3", "C4")):
    fig, axes = plt.subplots(1, len(channels), figsize=(11, 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ch in zip(axes, channels):
        for cls, name in CLASS_NAMES.items():
            try:
                e = epochs_clean[name].copy().pick([ch]).filter(
                    8, 30, verbose=False)
            except Exception:
                continue
            power = (e.get_data()[:, 0, :] ** 2).mean(axis=0)
            ax.plot(e.times, power, color=CLASS_COLORS[cls], label=name, lw=1.1)
        ax.axvline(0, color=MUTED, ls="--", lw=1)
        ax.set_title(f"{ch} · band power 8-30 Hz")
        ax.set_xlabel("time from cue (s)")
    axes[0].set_ylabel("power")
    axes[0].legend(fontsize=7)
    _dark(fig, axes)
    fig.tight_layout()
    return fig

# Comparison
def plot_comparison(df, chance=0.25):
    if df is None or len(df) == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(.5, .5, "no results", ha="center", color=INK)
        ax.axis("off"); _dark(fig, ax)
        return fig

    pivot = df.pivot_table(index="feature_method", columns="classifier",
                           values="mean_accuracy")
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(3.4, .5 * len(pivot) + 2)))

    # heatmap
    ax = axes[0]
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto",
                   vmin=max(0, chance - .05),
                   vmax=max(.5, np.nanmax(pivot.values)))
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Mean accuracy")
    best = np.nanmax(pivot.values)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if v < best * .8 else "black",
                    fontweight="bold" if np.isclose(v, best) else "normal")
    cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    cb.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=MUTED)

    # ranked bars
    ax2 = axes[1]
    d = df.copy()
    d["combo"] = d["feature_method"] + " · " + d["classifier"]
    d = d.sort_values("mean_accuracy")
    bars = ax2.barh(np.arange(len(d)), d["mean_accuracy"],
                    xerr=d["std_accuracy"], color=SIGNAL, capsize=2,
                    error_kw={"ecolor": MUTED})
    if len(bars):
        bars[-1].set_color(AMBER)
    ax2.set_yticks(np.arange(len(d)))
    ax2.set_yticklabels(d["combo"], fontsize=7)
    ax2.axvline(chance, color=AMBER, ls="--", lw=1, label=f"chance {chance:.2f}")
    ax2.set_xlabel("accuracy (± std)")
    ax2.legend(fontsize=7)
    _dark(fig, axes)
    fig.tight_layout()
    return fig



