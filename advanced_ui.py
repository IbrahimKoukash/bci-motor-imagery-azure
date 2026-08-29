"""
Renders the Advanced-Settings tab from the param registry.

Only shows controls for the methods the user actually selected (passed in),
so the tab stays clean. Returns a params dict shaped for decoding_params.
Store the result in session_state and pass to run_comparison / evaluate_p.

"""
import streamlit as st
from params import EXTRACTOR_PARAMS, CLASSIFIER_PARAMS, default_params


def _control(scope, name, spec):
    key, label, kind, default, meta = spec
    wkey = f"param_{scope}_{name}_{key}"
    if kind == "slider_int":
        lo, hi, step = meta
        return st.slider(label, lo, hi, default, step, key=wkey)
    if kind == "slider_float":
        lo, hi, step = meta
        return st.slider(label, float(lo), float(hi), float(default),
                         float(step), key=wkey)
    if kind == "select":
        # meta is a list of options
        idx = meta.index(default) if default in meta else 0
        return st.selectbox(label, meta, index=idx, key=wkey,
                            format_func=lambda v: f"{v}")
    return default


def render_advanced(selected_extractors, selected_classifiers):
    """
    Render controls only for selected methods; return a params dict.
    """
    params = default_params()

    st.markdown("Parameters are **set once and fixed** across the whole "
                "comparison, so every feature x classifier is judged on the "
                "same footing. Only the methods you selected in step 02 appear "
                "here.")

    # Feature extractor params 
    st.markdown("#### Feature extractors")
    any_ex = False
    for name in selected_extractors:
        specs = EXTRACTOR_PARAMS.get(name)
        if not specs:
            continue
        any_ex = True
        with st.expander(f"{name}", expanded=False):
            vals = {}
            cols = st.columns(min(2, len(specs)))
            for i, spec in enumerate(specs):
                with cols[i % len(cols)]:
                    vals[spec[0]] = _control("ex", name, spec)
            params["extractors"][name] = vals
    if not any_ex:
        st.caption("None of the selected extractors expose tunable "
                   "parameters (variance, Hjorth, entropy, wavelet use fixed "
                   "settings).")

    # Classifier params 
    st.markdown("#### Classifiers")
    any_cl = False
    for name in selected_classifiers:
        specs = CLASSIFIER_PARAMS.get(name)
        if not specs:
            continue
        any_cl = True
        with st.expander(f"{name}", expanded=False):
            vals = {}
            cols = st.columns(min(2, len(specs)))
            for i, spec in enumerate(specs):
                with cols[i % len(cols)]:
                    vals[spec[0]] = _control("cl", name, spec)
            params["classifiers"][name] = vals
    if not any_cl:
        st.caption("None of the selected classifiers expose tunable "
                   "parameters (LDA has none).")

    return params
