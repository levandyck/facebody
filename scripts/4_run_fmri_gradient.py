"""Step 4: Build ventral posterior-to-anterior axis and measure selectivity along it."""

from facebody.config import PROJECT_ROOT, SUBJECTS
from facebody.fmri.gradient import build_gradient, run_fmri_profile
from facebody.myutils.nsd import SubjectLoader

# Build posterior-anterior gradient
for subj in SUBJECTS:
    sl = SubjectLoader(subj, PROJECT_ROOT)
    build_gradient(subj, sl, PROJECT_ROOT)

# Measure selectivity along gradient
run_fmri_profile(SUBJECTS, PROJECT_ROOT)
