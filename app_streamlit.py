"""
BCIC IV 2a - pipeline app, polished UI.

Visual direction: a lab instrument for reading brain signals. Deep slate
ground, signal-cyan accent (oscilloscope trace), monospace for data/labels.
The two-step workflow is numbered because the pipeline genuinely is a
sequence: you must preprocess before you can decode.

Logic modules (pipeline / decoding / comparison / plots / storage) are
unchanged — this file is only the interface.

"""
import os
import tempfile
from pathlib import Path

import streamlit as st

# Example sessions bundled into the image 
EXAMPLE_DIR = Path(__file__).resolve().parent / "examples"

def example_files():
    """Bundled subject code -> path, e.g. {'A01T': PosixPath(....)}."""
    if not EXAMPLE_DIR.is_dir():
        return {}
    return {p.stem: p for p in sorted(EXAMPLE_DIR.glob("*.mat"))}


from storage import upload_bytes, BLOB_ENABLED, UPLOADS, ACCOUNT_URL
from pipeline import preprocess
from decoding import EXTRACTORS, CLASSIFIERS
from comparison import run_comparison
from advanced_ui import render_advanced
from plots import plot_psd, plot_erd, plot_comparison
from flow import flow_svg, PREPROCESS_STAGES, DECODE_STAGES
from viz_extra import (plot_ica_eog, plot_ica_topographies, plot_erd_topomaps,
                       plot_confusion, plot_montage)
from sweep_tab import (available_subjects, run_sweep, save_sweep_to_blob,
                       plot_best_per_subject, plot_subject_feature_heatmap)
from decoding_algorithms import evaluate_full # accuracy + confusion matrix

# page + theme
st.set_page_config(page_title=" CCAMIC Neuro Decode · BCI IV 2a",
                   page_icon="◐", layout="wide",
                   initial_sidebar_state="expanded")

CSS = """
<style>
/* ---- palette ---------------------------------------------------------- */
:root{
  --ground:#0E1621; --panel:#16212E; --panel-2:#1D2A3A;
  --ink:#E6EEF5; --muted:#8AA0B4; --line:#26374A;
  --signal:#31E0C8;         /* oscilloscope cyan — the accent */
  --signal-dim:#1C8C7E;
  --amber:#E7B45A;          /* 'best result' highlight only */
}
/* ---- base ------------------------------------------------------------- */
.stApp{background:
   radial-gradient(1200px 400px at 15% -5%, #17293b 0%, transparent 60%),
   var(--ground); color:var(--ink);}
section.main > div{padding-top:1.2rem;}
h1,h2,h3,h4{color:var(--ink); letter-spacing:-.01em;}
p,label,span,div{color:var(--ink);}
.stMarkdown small,.stCaption,.st-emotion-cache-1wivap2{color:var(--muted)!important;}

/* monospace for data-ish things */
code,.mono{font-family:"JetBrains Mono","SF Mono",Menlo,monospace;}

/* ---- masthead --------------------------------------------------------- */
.mast{border-bottom:1px solid var(--line); padding:.2rem 0 1rem; margin-bottom:1.2rem;}
.mast .kicker{font-family:"JetBrains Mono",monospace; font-size:.72rem;
  letter-spacing:.28em; text-transform:uppercase; color:var(--signal);}
.mast h1{font-size:2.0rem; margin:.15rem 0 .1rem; font-weight:650;}
.mast .sub{color:var(--muted); font-size:.95rem;}
.wave{height:34px; width:100%; margin-top:.5rem; opacity:.9;}

/* ---- step chips ------------------------------------------------------- */
.step{font-family:"JetBrains Mono",monospace; font-size:.72rem;
  letter-spacing:.18em; text-transform:uppercase; color:var(--muted);
  border:1px solid var(--line); border-radius:999px; padding:.28rem .7rem;
  display:inline-block;}
.step b{color:var(--signal);}

/* ---- panels ----------------------------------------------------------- */
.panel{background:var(--panel); border:1px solid var(--line);
  border-radius:14px; padding:1.1rem 1.2rem;}
.panel-h{font-family:"JetBrains Mono",monospace; font-size:.72rem;
  letter-spacing:.2em; text-transform:uppercase; color:var(--muted);
  margin-bottom:.6rem;}

/* ---- metric card ------------------------------------------------------ */
.metric{background:linear-gradient(180deg,var(--panel-2),var(--panel));
  border:1px solid var(--line); border-left:3px solid var(--signal);
  border-radius:12px; padding:.9rem 1.1rem;}
.metric .k{font-family:"JetBrains Mono",monospace; font-size:.7rem;
  letter-spacing:.18em; text-transform:uppercase; color:var(--muted);}
.metric .v{font-size:1.9rem; font-weight:680; font-family:"JetBrains Mono",monospace;
  color:var(--signal); line-height:1.1;}
.metric.best{border-left-color:var(--amber);}
.metric.best .v{color:var(--amber);}

/* ---- buttons ---------------------------------------------------------- */
.stButton>button{background:var(--signal); color:#04120F; border:0;
  border-radius:10px; font-weight:650; letter-spacing:.02em; padding:.55rem 1rem;}
.stButton>button:hover{background:#5CEBD8; color:#04120F;}

/* inputs a touch darker to sit on panels */
.stSelectbox,.stMultiSelect,.stNumberInput,.stSlider{color:var(--ink);}
[data-baseweb="tag"]{background:var(--signal-dim)!important;}

/* tabs */
.stTabs [data-baseweb="tab-list"]{gap:.3rem; border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{font-family:"JetBrains Mono",monospace;
  letter-spacing:.12em; text-transform:uppercase; font-size:.74rem;
  color:var(--muted);}
.stTabs [aria-selected="true"]{color:var(--signal)!important;}

/* dataframe */
.stDataFrame{border:1px solid var(--line); border-radius:10px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# inline SVG icons (inherit theme colors via currentColor) 
# small, stroke-based, sized to sit inline with the monospace panel labels.

_ICON_PATHS = {
    "input": 'M4 4h16v4H4zM4 12h10v4H4z', # tray/stack
    "filter": 'M3 5h18l-7 8v6l-4-2v-4z', # funnel
    "window": 'M4 5h16v14H4zM4 9h16', # framed box
    "signal": 'M2 12h4l3-7 4 14 3-9 2 2h4', # waveform
    "brain": 'M12 3a4 4 0 0 0-4 4v10a4 4 0 0 0 8 0V7a4 4 0 0 0-4-4zM12 3v18M8 8h8M8 13h8', # head/hemispheres
    "features":'M4 18V9M9 18V4M14 18v-7M19 18v-11', # bars
    "clf": 'M12 4v4m0 0l-5 4m5-4l5 4M7 12v4m5 0v4m5-4v4', # decision tree
    "chart": 'M4 20V4M4 20h16M8 16l3-4 3 2 4-6', # line chart
    "grid": 'M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z', # matrix
    "run": 'M6 4l14 8-14 8z', # play
    "download":'M12 3v12m0 0l-4-4m4 4l4-4M4 21h16', # down arrow
    "session": 'M12 2a5 5 0 0 0-5 5c0 3 5 9 5 9s5-6 5-9a5 5 0 0 0-5-5z', # pin
}


def icon(name, size=15, color="var(--signal)", stroke=1.7):
    d = _ICON_PATHS.get(name, "")
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" '
            f'fill="none" stroke="{color}" stroke-width="{stroke}" '
            f'stroke-linecap="round" stroke-linejoin="round" '
            f'style="vertical-align:-2px;margin-right:6px">'
            f'<path d="{d}"/></svg>')


def panel_h(label, ic=None, mt=False):
    """
    Themed panel header with an optional leading icon.
    """
    style = 'margin-top:1rem' if mt else ''
    ico = icon(ic, color="var(--muted)") if ic else ''
    st.markdown(f'<div class="panel-h" style="{style}">{ico}{label}</div>',
                unsafe_allow_html=True)

# an EEG waveform for the masthead (four class-colored traces)
WAVE_SVG = """
<svg class="wave" viewBox="0 0 1200 40" preserveAspectRatio="none">
  <polyline
    fill="none"
    stroke="#31E0C8"
    stroke-width="1.6"
    stroke-linecap="round"
    stroke-linejoin="round"
    opacity="0.95"
    points="
0,20
15,19 30,21 45,18 60,20 75,17
90,22 105,18 120,19 135,16 150,22
165,15 180,25 192,10 202,31 215,17
230,19 245,18 260,20 275,16 290,22
305,18 320,20 335,17 350,23 365,18
380,19 395,16 410,21 425,18 440,24
455,17 470,19 485,18 500,20 515,15
530,23 545,17 560,20 575,16 590,22
605,19 620,18 635,21 650,17 665,26
678,9 690,30 705,18 720,20 735,17
750,22 765,18 780,19 795,15 810,23
825,18 840,20 855,17 870,24 885,18
900,21 915,16 930,22 945,18 960,20
975,17 990,25 1002,11 1015,29 1030,18
1045,20 1060,17 1075,22 1090,18 1105,20
1120,17 1135,21 1150,18 1165,20 1180,19 1200,20
"/>
</svg>
"""

# masthead
st.markdown(f"""
<div class="mast">
  <div class="kicker">Motor-Imagery Decoding · BCI Competition IV · Dataset 2a</div>
  <h1>CCAMIC Neuro Decode</h1>
  <div class="sub">Load a session, clean the signal, and compare feature
   extractors and classifiers — four movements: left hand, right hand, feet, tongue.</div>
  {WAVE_SVG}
</div>
""", unsafe_allow_html=True)

# session store
if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.subject = None
if "params" not in st.session_state:
    from params import default_params
    st.session_state.params = default_params()

# storage status in the sidebar
with st.sidebar:
    import os as _os
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _kyushu = _os.path.join(_here, "assets", "kyushu_logo_dark.png")
    _lab = _os.path.join(_here, "assets", "lab_logo.png")
    if _os.path.exists(_kyushu):
        st.image(_kyushu, use_container_width=True)
    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

    st.markdown('<div class="panel-h">' +
                icon("session", color="var(--muted)") +
                'Session</div>', unsafe_allow_html=True)
    if BLOB_ENABLED:
        acct = ACCOUNT_URL.split("//")[-1].split(".")[0]
        st.markdown(f"Storage: **connected**\n\n`{acct}`")
    else:
        st.markdown("Storage: **local only**\n\nResults stay in this session.")
    if st.session_state.result:
        r = st.session_state.result
        st.markdown(f"Loaded: **{st.session_state.subject}**")
        st.markdown(f"`{r['epochs_data'].shape[0]} trials · "
                    f"{r['epochs_data'].shape[1]} ch`")
    else:
        st.markdown("Loaded: **—**")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    if _os.path.exists(_lab):
        st.image(_lab, use_container_width=True)
    st.caption("Dept. of Mechanical Engineering · System Engineering Lab")

tab1, tab2, tab3, tab4 = st.tabs(["①  Preprocess", "②  Decode & compare",
                                  "⚙  Advanced settings",
                                  "▦  Cross-subject sweep"])

# Tab 1
with tab1:
    st.markdown('<span class="step">Step <b>01</b> — clean the signal</span>',
                unsafe_allow_html=True)
    st.write("")
 
    # pipeline flow strip - lights up as preprocessing advances
    flow1 = st.empty()
    if st.session_state.result is not None:
        flow1.markdown(flow_svg(PREPROCESS_STAGES, 1.0), unsafe_allow_html=True)
    else:
        flow1.markdown(flow_svg(PREPROCESS_STAGES, 0.0), unsafe_allow_html=True)
    st.write("")
 
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        panel_h("Input", "input")
        _examples = example_files()
        if _examples:
            src = st.radio("Data source",
                           ["Example subject", "Upload .mat"],
                           horizontal=True, label_visibility="collapsed")
        else:
            src = "Upload .mat"
 
        up = None
        example_name = None
        if src == "Example subject":
            example_name = st.selectbox("Bundled subject", list(_examples),
                                        label_visibility="collapsed")
            st.caption("Bundled with the app — BCI Competition IV 2a "
                       "BCI Competition IV (Dataset 2a),Institute for Knowledge Discovery, "
                       "Graz University of Technology, 2008")
        else:
            up = st.file_uploader("Session file (.mat)", type=["mat"],
                                  label_visibility="collapsed")
            st.caption("BCIC IV 2a training file, e.g. A01T.mat")
 
        panel_h("Filter", "filter", mt=True)
        c1, c2 = st.columns(2)
        l_freq = c1.number_input("High-pass (Hz)", value=1.0, step=0.5)
        h_freq = c2.number_input("Low-pass (Hz)", value=40.0, step=5.0)
 
        st.markdown('<div class="panel-h" style="margin-top:1rem">'
                    'Motor-imagery window (s from cue)</div>',
                    unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        erd_lo = c3.number_input("from", value=0.5, step=0.1)
        erd_hi = c4.number_input("to", value=2.5, step=0.1)
 
        do_ica = st.checkbox("Remove eye artifacts (ICA) — adds ~60 s",
                             value=True)
        run = st.button("▶  Run preprocessing", type="primary",
                        use_container_width=True)
 
    with right:
        panel_h("Signal readout", "signal")
        readout = st.container()

    if run:
        # both sources collapse to a real path, so preprocess() sees one shape
        mat_path, subject, raw_bytes = None, None, None
        if example_name is not None:
            mat_path = str(_examples[example_name])
            subject = example_name
        elif up is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mat")
            tmp.write(up.getbuffer()); tmp.close()
            mat_path = tmp.name
            subject = up.name.replace(".mat", "")
            raw_bytes = up.getvalue()
 
        if mat_path is None:
            st.warning("Choose a .mat session file first.")
        else:
            # only archive genuine uploads; the bundled example is already
            # in the image and re-uploading it every run is pure noise
            if BLOB_ENABLED and raw_bytes is not None:
                try:
                    upload_bytes(UPLOADS, up.name, raw_bytes)
                except Exception as e:
                    st.info(f"Upload archive skipped — {e}")
 
            prog = st.progress(0.0, text="starting")
 
            def _cb(frac, desc=""):
                prog.progress(min(1.0, float(frac)), text=desc)
                flow1.markdown(flow_svg(PREPROCESS_STAGES, float(frac)),
                               unsafe_allow_html=True)
 
            try:
                res = preprocess(mat_path, l_freq=l_freq, h_freq=h_freq,
                                 erd_tmin=erd_lo, erd_tmax=erd_hi,
                                 do_ica=do_ica, progress=_cb)
            except Exception as e:
                st.error(f"Preprocessing stopped: {e}")
                res = None
 
            if res is not None:
                prog.progress(1.0, text="done")
                res["subject"] = subject
                st.session_state.result = res
                st.session_state.subject = subject
                d = res["epochs_data"]
 
                with readout:
                    m1, m2, m3 = st.columns(3)
                    m1.markdown(f'<div class="metric"><div class="k">Trials</div>'
                                f'<div class="v">{d.shape[0]}</div></div>',
                                unsafe_allow_html=True)
                    m2.markdown(f'<div class="metric"><div class="k">Channels</div>'
                                f'<div class="v">{d.shape[1]}</div></div>',
                                unsafe_allow_html=True)
                    m3.markdown(f'<div class="metric"><div class="k">EOG removed</div>'
                                f'<div class="v">{res["n_eog_removed"]}</div></div>',
                                unsafe_allow_html=True)
                    st.write("")
                    st.caption(f"MI runs {res['mi_runs']} · classes "
                               f"{res['class_counts']} · "
                               f"{res['sfreq']:.0f} Hz · window {res['erd_window']} s")
                    # electrode montage — where the channels sit
                    try:
                        st.pyplot(plot_montage(res["epochs_clean"].info))
                        st.caption("The 22 EEG electrodes for this recording, "
                                   "in standard 10-20 positions. C3 and C4 (over "
                                   "left/right motor cortex) are the key hand-"
                                   "imagery channels.")
                    except Exception as e:
                        st.caption(f"Montage unavailable: {e}")
 
                    st.pyplot(plot_psd(res["raw_filt"]))
                    st.caption("Signal power at each frequency after filtering. "
                               "The band-pass keeps 1-40 Hz, where motor-imagery "
                               "rhythms live, and removes slow drift and noise.")
                    st.pyplot(plot_erd(res["epochs_clean"]))
                    st.caption("Band power over time on the two hand-motor "
                               "channels (C3, C4). A power drop after the cue "
                               "(dashed line) is the motor-imagery signal - the "
                               "brain rhythm desynchronizing during imagined "
                               "movement.")
 
                    # eye-artifact removal transparency 
                    st.markdown('<div class="panel-h" style="margin-top:1rem">'
                                'Eye-artifact removal (ICA)</div>',
                                unsafe_allow_html=True)
                    if res.get("ica") is not None:
                        st.pyplot(plot_ica_eog(
                            res["ica"], res["epochs_clean"],
                            res.get("eog_idx", []), res.get("eog_scores", [])))
                        st.caption("Independent components whose time-course "
                                   "matches the EOG (eye) channels are flagged "
                                   "as blinks/eye-movements and removed. Amber "
                                   "bars are the removed components; the % is "
                                   "how strongly each tracked the eye signal.")
                        topo = plot_ica_topographies(res["ica"],
                                                     res.get("eog_idx", []))
                        if topo is not None:
                            with st.expander("Removed-component topographies"):
                                st.pyplot(topo)
                    else:
                        st.caption("ICA was skipped for this run.")
 
                    # where each movement shows up 
                    st.markdown('<div class="panel-h" style="margin-top:1rem">'
                                'Where each movement shows up '
                                '(band-power topography)</div>',
                                unsafe_allow_html=True)
                    try:
                        st.pyplot(plot_erd_topomaps(res["epochs_clean"]))
                        st.caption("Scalp maps of mu/beta band power for each "
                                   "movement. Hand imagery should activate the "
                                   "motor cortex on the opposite side of the "
                                   "head (left hand → right side, and vice "
                                   "versa). Blue = lower power (desynchronization).")
                    except Exception as e:
                        st.caption(f"Topomaps unavailable: {e}")

# Tab 2
with tab2:
    st.markdown('<span class="step">Step <b>02</b> — decode & compare</span>',
                unsafe_allow_html=True)
    st.write("")

    flow2 = st.empty()
    flow2.markdown(flow_svg(DECODE_STAGES, 0.0), unsafe_allow_html=True)
    st.write("")

    if st.session_state.result is None:
        st.info("Run preprocessing in step 01 first — the cleaned signal "
                "feeds the decoders here.")
    else:
        left, right = st.columns([1, 1.7], gap="large")
        with left:
            panel_h("Feature extractors", "features")
            feats = st.multiselect("features", list(EXTRACTORS.keys()),
                                   default=["freq_bandpower", "spatial_csp"],
                                   label_visibility="collapsed")
            combine = st.checkbox("Also test all selected combined")

            st.markdown('<div class="panel-h" style="margin-top:1rem">'
                        'Classifiers</div>', unsafe_allow_html=True)
            clfs = st.multiselect("classifiers", list(CLASSIFIERS.keys()),
                                  default=["lda", "svm"],
                                  label_visibility="collapsed")
            # remember selections so the Advanced tab shows only these
            st.session_state.sel_feats = feats
            st.session_state.sel_clfs = clfs

            st.markdown('<div class="panel-h" style="margin-top:1rem">'
                        'Windowing</div>', unsafe_allow_html=True)
            win = st.slider("Window (s)", 0.5, 2.0, 2.0, 0.1)
            ovl = st.slider("Overlap (%)", 0, 75, 0, 25)
            st.caption("Full window · 0 % overlap → one window per trial "
                       "(best for CSP).")

            cmp = st.button("▶  Run comparison", type="primary",
                            use_container_width=True)

        with right:
            out = st.container()

        if cmp:
            if not feats or not clfs:
                st.warning("Pick at least one feature extractor and one classifier.")
            else:
                feature_sets = [[f] for f in feats]
                if combine and len(feats) > 1:
                    feature_sets.append(list(feats))

                prog2 = st.progress(0.0, text="scoring")

                def _cb2(frac, desc=""):
                    prog2.progress(min(1.0, float(frac)), text=desc)
                    flow2.markdown(flow_svg(DECODE_STAGES, float(frac)),
                                   unsafe_allow_html=True)

                df, blob_path = run_comparison(
                    st.session_state.result, feature_sets, clfs, win, ovl,
                    subject=st.session_state.subject, progress=_cb2)
                prog2.progress(1.0, text="done")

                with out:
                    if df is None or len(df) == 0:
                        st.error("No results — every configuration failed to run.")
                    else:
                        b = df.iloc[0]
                        cc1, cc2 = st.columns([1, 1])
                        cc1.markdown(
                            f'<div class="metric best"><div class="k">Best combination</div>'
                            f'<div class="v">{b["mean_accuracy"]*100:.1f}%</div>'
                            f'<div class="k" style="margin-top:.3rem">'
                            f'{b["feature_method"]} · {b["classifier"]}</div></div>',
                            unsafe_allow_html=True)
                        cc2.markdown(
                            f'<div class="metric"><div class="k">Chance level</div>'
                            f'<div class="v">25.0%</div>'
                            f'<div class="k" style="margin-top:.3rem">'
                            f'4 classes · {b["n_trials"]} trials</div></div>',
                            unsafe_allow_html=True)
                        if blob_path:
                            st.caption(f"Saved → {blob_path}")
                        st.write("")
                        st.pyplot(plot_comparison(df))
                        st.caption("Left: accuracy for every feature × "
                                   "classifier pairing (brighter = better, best "
                                   "boxed). Right: same results ranked, with "
                                   "the chance line (25% for four classes).")
                        panel_h("All results", "grid")
                        st.dataframe(df, use_container_width=True, height=280)

                        # download results as CSV
                        st.download_button(
                            "↓  Download results (CSV)",
                            data=df.to_csv(index=False).encode("utf-8"),
                            file_name=f"results_{st.session_state.subject}.csv",
                            mime="text/csv")

                        # confusion matrix for the best combo 
                        st.markdown('<div class="panel-h" style="margin-top:1rem">'
                                    'How the best model confuses classes</div>',
                                    unsafe_allow_html=True)
                        try:
                            best_feats = b["feature_method"].split("+")
                            r = evaluate_full(
                                st.session_state.result["epochs_data"],
                                st.session_state.result["labels"],
                                st.session_state.result["sfreq"],
                                win, ovl, best_feats,
                                classifier=b["classifier"])
                            st.pyplot(plot_confusion(
                                r["confusion_matrix"], r["classes"],
                                title=f"{b['feature_method']} · "
                                      f"{b['classifier']}  "
                                      f"(acc={r['accuracy']:.2f})"))
                            st.caption("Rows are the true movement, columns are "
                                       "what the model predicted. A strong "
                                       "diagonal means correct classifications; "
                                       "off-diagonal cells show which movements "
                                       "get confused with each other.")
                        except Exception as e:
                            st.caption(f"Confusion matrix unavailable: {e}")

# Tab 3
with tab3:
    st.markdown('<span class="step">Advanced — <b>tune parameters</b></span>',
                unsafe_allow_html=True)
    st.write("")
    sel_feats = st.session_state.get("sel_feats",
                                     ["freq_bandpower", "spatial_csp"])
    sel_clfs = st.session_state.get("sel_clfs", ["lda", "svm"])
    st.session_state.params = render_advanced(sel_feats, sel_clfs)
    st.caption("Changes apply on the next 'Run comparison' in step 02.")

# Tab 4 
with tab4:
    st.markdown('<span class="step">Cross-subject — <b>compare across '
                'people</b></span>', unsafe_allow_html=True)
    st.write("")

    subs = available_subjects()
    if not subs:
        st.info("No cached subjects found. Run `cache_subjects.py` once to "
                "pre-process the subjects, then bundle the `cache/` folder "
                "with the app. Preprocessing is done ahead of time so this "
                "sweep stays fast.")
    else:
        st.caption(f"{len(subs)} subjects pre-processed and ready. Pick what "
                   "to sweep — preprocessing is already done, so only the fast "
                   "decoding runs here.")
        left, right = st.columns([1, 1.7], gap="large")
        with left:
            panel_h("Subjects", "session")
            sw_subs = st.multiselect("subjects", subs,
                                     default=subs[:min(4, len(subs))],
                                     label_visibility="collapsed")
            panel_h("Feature extractors", "features", mt=True)
            sw_feats = st.multiselect("sweep_features", list(EXTRACTORS.keys()),
                                      default=["spatial_csp", "freq_bandpower"],
                                      label_visibility="collapsed")
            sw_combine = st.checkbox("Also test selected features combined")
            panel_h("Classifiers", "clf", mt=True)
            sw_clfs = st.multiselect("sweep_clfs", list(CLASSIFIERS.keys()),
                                     default=["lda", "svm"],
                                     label_visibility="collapsed")
            panel_h("Windowing", "window", mt=True)
            sw_win = st.slider("Window (s)", 0.5, 2.0, 2.0, 0.1, key="sw_win")
            sw_ovl = st.slider("Overlap (%)", 0, 75, 0, 25, key="sw_ovl")

            n_jobs = (len(sw_subs) * (len(sw_feats) + (1 if sw_combine else 0))
                      * len(sw_clfs))
            st.caption(f"≈ {n_jobs} model fits — bounded by your choices.")
            sw_run = st.button("▶  Run cross-subject sweep", type="primary",
                               use_container_width=True)

        with right:
            sw_out = st.container()

        if sw_run:
            if not sw_subs or not sw_feats or not sw_clfs:
                st.warning("Pick at least one subject, feature, and classifier.")
            else:
                fsets = [[f] for f in sw_feats]
                if sw_combine and len(sw_feats) > 1:
                    fsets.append(list(sw_feats))
                pbar = st.progress(0.0, text="sweeping")

                def _swcb(frac, desc=""):
                    pbar.progress(min(1.0, float(frac)), text=desc)

                dfw = run_sweep(sw_subs, fsets, sw_clfs, sw_win, sw_ovl,
                                params=st.session_state.params, progress=_swcb)
                pbar.progress(1.0, text="done")

                with sw_out:
                    if dfw is None or len(dfw) == 0:
                        st.error("No results — every configuration failed.")
                    else:
                        best = dfw.loc[dfw["mean_accuracy"].idxmax()]
                        st.markdown(
                            f'<div class="metric best"><div class="k">Top result'
                            f'</div><div class="v">{best["mean_accuracy"]*100:.1f}%'
                            f'</div><div class="k" style="margin-top:.3rem">'
                            f'{best["subject"]} · {best["feature_method"]} · '
                            f'{best["classifier"]}</div></div>',
                            unsafe_allow_html=True)
                        st.write("")
                        st.pyplot(plot_best_per_subject(dfw))
                        st.caption("The best accuracy reached for each subject, "
                                   "with the winning feature·classifier labeled. "
                                   "Cross-subject variability is expected in "
                                   "motor imagery — some people decode far "
                                   "better than others.")
                        st.pyplot(plot_subject_feature_heatmap(dfw))
                        st.caption("Average accuracy per subject and feature "
                                   "method (across the chosen classifiers).")
                        st.dataframe(dfw.sort_values("mean_accuracy",
                                                     ascending=False),
                                     use_container_width=True, height=260)
                        st.download_button(
                            "↓  Download sweep (CSV)",
                            data=dfw.to_csv(index=False).encode("utf-8"),
                            file_name="cross_subject_sweep.csv", mime="text/csv")

                        # persist to Blob so this run reaches the Fabric
                        # Lakehouse. Never let a storage failure break the
                        if BLOB_ENABLED:
                            try:
                                bp = save_sweep_to_blob(dfw,
                                                        user_id="reviewer")
                                st.caption(f"Saved → {bp} · this run "
                                           "will appear in the Fabric report.")
                            except Exception as e:
                                st.caption(f"Not saved to Blob — {e}")
                        else:
                            st.caption("Blob storage is off — results are "
                                       "download-only for this session.")
                        
                        