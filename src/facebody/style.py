"""Matplotlib defaults."""

import matplotlib.pyplot as plt

plt.rcParams.update({
    "axes.linewidth": 1,
    "lines.linewidth": 2.5,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.title_fontsize": 13,
    "legend.fontsize": 12,
    "figure.titlesize": 13,
    "figure.dpi": 300
})

SEL_INFO = {
    "face": {"label": "Face-selective", "color": "#dc267fff"},
    "body": {"label": "Body-selective", "color": "#ffb000ff"},
    "mixed": {"label": "Mixed-selective", "color": "#648fffff"},
    "nonselective":{"label": "Non-selective", "color": "#ccccccff"},
}

CTRL_INFO = {
    "scene": {"label": "Scene-selective", "color": "#785ef0ff"},
}

UNIT_INFO = {**SEL_INFO, **CTRL_INFO}

FACE_COLOR = SEL_INFO["face"]["color"][:7]
BODY_COLOR = SEL_INFO["body"]["color"][:7]
MIXED_COLOR = SEL_INFO["mixed"]["color"][:7]
NON_COLOR = SEL_INFO["nonselective"]["color"][:7]

SHARED_COLOR = "#F9CEB0"
SHARED_COLOR_FLATMAP = "#ffb07f"
