"""Step 6: Train each task readout, lesion each unit type, and measure accuracy drop."""

from facebody.model_sets import ENCODING_MODEL, LESIONING_LAYER, LESIONING_TASKS
from facebody.dnn.lesioning import run_lesioning_analysis

device = "cuda:0"

# Run lesioning analysis
for task in LESIONING_TASKS:
    run_lesioning_analysis(
        ENCODING_MODEL, LESIONING_LAYER, task, device,
    )
