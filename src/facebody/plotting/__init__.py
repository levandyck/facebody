"""Figures and statistics, one module per figure family."""

from facebody.style import SEL_INFO

from .common import (
    _mean_err, compute_mean_ci, compute_mean_ci_hierarch_models_layers_units,
    compute_mean_ci_within,
)
from .selectivity import (
    plot_perc_layers, stats_perc_layers,
    plot_perc_layers_by_model, plot_perc_layers_mixed,
    plot_dprime_dist, stats_dprime, stats_dprime_dd,
    plot_resp_dist, plot_dprime_layers,
    plot_perc_layers_by_diet,
)
from .gradcam import plot_guided_gradcam, plot_guided_gradcam_diff, plot_top_nsd_imgs
from .encoding import (
    plot_encoding_sep, plot_encoding_varpart,
    plot_encoding_sep_layers, plot_encoding_varpart_layers,
)
from .ventral import (
    _encoding_ventral_curves, plot_perc_ventral, stats_perc_ventral,
    plot_sep_ventral, plot_varpart_ventral,
    plot_integration_ventral, stats_integration_ventral,
)
from .lesioning import plot_lesioning_drop

__all__ = [
    "SEL_INFO",
    "_mean_err",
    "compute_mean_ci",
    "compute_mean_ci_hierarch_models_layers_units",
    "compute_mean_ci_within",

    "plot_perc_layers",
    "stats_perc_layers",
    "plot_perc_layers_by_model",
    "plot_perc_layers_mixed",
    "plot_dprime_dist",
    "stats_dprime",
    "stats_dprime_dd",
    "plot_resp_dist",
    "plot_dprime_layers",
    "plot_guided_gradcam",
    "plot_guided_gradcam_diff",
    "plot_perc_layers_by_diet",

    "plot_perc_ventral",
    "stats_perc_ventral",

    "plot_top_nsd_imgs",
    "plot_encoding_sep",
    "plot_encoding_varpart",
    "plot_encoding_sep_layers",
    "plot_encoding_varpart_layers",
    "_encoding_ventral_curves",
    "plot_sep_ventral",
    "plot_varpart_ventral",
    "plot_integration_ventral",
    "stats_integration_ventral",

    "plot_lesioning_drop",
]
