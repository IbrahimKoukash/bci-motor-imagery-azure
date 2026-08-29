"""

Pipeline flow strip - boxes that light up as each stage completes.

Rendered as inline SVG so it restyles with the app theme. Driven by the same
progress callback the pipeline already emits, so a box switches from 'pending'
to 'active' to 'done' as the run advances. Import and use from the app.

"""

# stages in order; the fraction is the progress value at which each becomes
# "done" (matches the _p() calls inside pipeline.preprocess)
PREPROCESS_STAGES = [
    ("load",   "Load",        0.20),
    ("filter", "Filter",      0.30),
    ("ref",    "Re-reference",0.35),
    ("ica",    "ICA",         0.75),
    ("epoch",  "Epoch",       0.88),
    ("crop",   "ERD crop",    0.95),
]

DECODE_STAGES = [
    ("window",  "Window",   0.10),
    ("feature", "Features", 0.35),
    ("scale",   "Scale",    0.55),
    ("cv",      "Cross-val",0.80),
    ("rank",    "Rank",     1.00),
]

_C = {
    "pending_fill": "#16212E", "pending_stroke": "#26374A", "pending_text": "#5F7488",
    "active_fill":  "#1D2A3A", "active_stroke":  "#E7B45A", "active_text":  "#E7B45A",
    "done_fill":    "#12332C", "done_stroke":    "#31E0C8", "done_text":    "#31E0C8",
    "arrow_off":    "#26374A", "arrow_on":       "#31E0C8",
}


def flow_svg(stages, frac):
    """
    
    Build an SVG flow strip. Boxes with threshold <= frac are 'done',
    the first box above frac is 'active', the rest 'pending'.

    """
    n = len(stages)
    box_w, box_h, gap = 132, 46, 34
    width = n * box_w + (n - 1) * gap
    height = 74

    # find the active index (first stage not yet done)
    active_idx = n
    for i, (_, _, thr) in enumerate(stages):
        if frac < thr:
            active_idx = i
            break

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
             f'preserveAspectRatio="xMidYMid meet" '
             f'xmlns="http://www.w3.org/2000/svg" '
             f'font-family="JetBrains Mono, monospace">']

    x = 0
    for i, (key, label, thr) in enumerate(stages):
        if frac >= thr:
            state = "done"
        elif i == active_idx:
            state = "active"
        else:
            state = "pending"
        fill = _C[f"{state}_fill"]; stroke = _C[f"{state}_stroke"]
        txt = _C[f"{state}_text"]
        cy = height / 2

        # box
        parts.append(
            f'<rect x="{x}" y="{cy-box_h/2}" width="{box_w}" height="{box_h}" '
            f'rx="10" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
        # a small status dot
        dot = {"done": "\u25CF", "active": "\u25D0", "pending": "\u25CB"}[state]
        parts.append(
            f'<text x="{x+16}" y="{cy+4}" fill="{txt}" font-size="13">{dot}</text>')
        # label
        parts.append(
            f'<text x="{x+34}" y="{cy+4}" fill="{txt}" font-size="12.5" '
            f'letter-spacing="1">{label}</text>')

        # arrow to next
        if i < n - 1:
            ax0 = x + box_w + 4
            ax1 = x + box_w + gap - 4
            on = frac >= thr           # arrow lights once this stage is done
            col = _C["arrow_on"] if on else _C["arrow_off"]
            parts.append(
                f'<line x1="{ax0}" y1="{cy}" x2="{ax1-6}" y2="{cy}" '
                f'stroke="{col}" stroke-width="1.6"/>')
            parts.append(
                f'<path d="M{ax1-6},{cy-4} L{ax1},{cy} L{ax1-6},{cy+4} Z" '
                f'fill="{col}"/>')
        x += box_w + gap

    parts.append("</svg>")
    return "".join(parts)
