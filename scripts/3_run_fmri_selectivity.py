"""Step 3: Find selective voxels using functional localizer GLM contrasts."""

from facebody.config import PROJECT_ROOT, SUBJECTS
from facebody.fmri.floc import run_fmri_selectivity

# Find selective voxels
run_fmri_selectivity(SUBJECTS, PROJECT_ROOT)
