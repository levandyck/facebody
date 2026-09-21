"""Step 2: Download NSD's fLoc runs and fit localizer GLM."""

from facebody.config import PROJECT_ROOT, SUBJECTS
from facebody.fmri.floc import FLOC_RUNS, download_floc_data, run_floc_glm

# Download NSD's preprocessed fLoc timeseries and BIDS events
download_floc_data(SUBJECTS, PROJECT_ROOT, FLOC_RUNS)

# Fit fLoc GLM
run_floc_glm(SUBJECTS, PROJECT_ROOT, FLOC_RUNS)
