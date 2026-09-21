"""Bar plot of lesioning accuracy drops."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from facebody.config import FIG_ROOT, PROJECT_ROOT
from facebody.style import SEL_INFO
from facebody.myutils.utils import clean_axes, load_pickle

# --------------------------------- Lesioning -------------------------------- #
def plot_lesioning_drop(model_name: str, tasks: tuple=("face", "person", "action"),
                        norm_drop: bool=True, fig_title: str=None, out_path: Path=None):
    """Bar plot: Lesioning drops for different classification tasks."""
    datasets = {
        "face": "Faces",
        "person": "Persons",
        "action": "Actions",
    }
    lims = {
        "face": [0, 30],
        "person": [0, 10],
        "action": [0, 10],
    }

    norm_key = "rel" if norm_drop else "abs"

    n_tasks = len(tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(2.5 * n_tasks, 3), squeeze=True)
    if n_tasks == 1:
        axes = [axes]

    sel_types = list(SEL_INFO.keys())
    colors = [SEL_INFO[s]["color"] for s in sel_types]
    x = np.arange(len(sel_types))

    for ax, task in zip(axes, tasks):
        res = load_pickle(PROJECT_ROOT / "lesioning" / model_name / task / "lesion_global.pkl")
        per_sel = res["summary"]["per_sel"]

        means = np.array([per_sel[s][norm_key]["mean_drop"] for s in sel_types])
        ci_lows = np.array([per_sel[s][norm_key]["ci_low"] for s in sel_types])
        ci_highs = np.array([per_sel[s][norm_key]["ci_high"] for s in sel_types])

        # Compute error bars: distance from mean to CI bounds
        lower_errs = means - ci_lows
        upper_errs = ci_highs - means
        errs = np.array([lower_errs, upper_errs])

        ax.bar(
            x,
            means * 100,
            yerr=errs * 100,
            capsize=2,
            color=colors,
            error_kw={"elinewidth": 1.5, "ecolor": "dimgray"},
        )
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set_ylim(lims[task])
        ax.set_xticks([])
        ax.set_ylabel("Norm. accuracy drop (%)" if norm_drop else "Accuracy drop (%)")
        ax.set_title(datasets[task])
        clean_axes(ax)

    fig.suptitle(fig_title)
    plt.tight_layout()
    if out_path:
        plt.savefig(FIG_ROOT / out_path, dpi=300)
    plt.show()
