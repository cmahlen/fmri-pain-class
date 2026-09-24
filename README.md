# fMRI crash course: interactive notebooks

Three Google Colab notebooks that follow the lecture (`fMRI_course_partI.pptx`). One subject, one 10-minute run
of painful heat to the lower back, from Baliki et al. 2010, *Neuron*, downloaded from the OpenPain repository (openpain.org, PDDL licence).

| Notebook | Lecture section | Live in Colab | Precomputed locally (FSL) |
|---|---|---|---|
| `notebooks/01_preprocessing.ipynb` [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmahlen/fmri-pain-class/blob/main/notebooks/01_preprocessing.ipynb) | slides 28-42 | temporal filtering, smoothing, all before/after views | motion correction (`mcflirt`), registration (`flirt`) |
| `notebooks/02_glm.ipynb` [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmahlen/fmri-pain-class/blob/main/notebooks/02_glm.ipynb) | slides 45-61 | HRF, one-voxel GLM, whole-brain GLM, thresholding, NAc onset/offset responses | |
| `notebooks/03_connectivity.ipynb` [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmahlen/fmri-pain-class/blob/main/notebooks/03_connectivity.ipynb) | slides 62-65 | seed maps, ROI correlation matrix, degree map, effect of preprocessing | |

Students need no Python. Every line they should touch is marked `# <-- change`.

## Setup

Students can open the notebooks with the Colab badge links above (they load straight from this GitHub repo). The setup cell installs nilearn and downloads ~110 MB.

## Rebuilding (not needed for students)

```bash
scripts/preprocess_fsl.sh              # FSL: motion correction, brain extraction, EPI -> T1 -> MNI (all 6 subjects, ~15 min)
scripts/make_bundle.sh cbp006 task001_run002   # build drive_upload/cbp006 from that run (default run001); tar it and upload to the data-v1 release (3 mm MNI bold as int16, raw, motion corrected, T1, mask, stimulus, rating)
python3 scripts/screen_subjects.py     # per-subject motion and NAc / insula response summary -> derivatives/screening.csv
python3 scripts/build_notebooks.py     # regenerate notebooks/*.ipynb from the cell sources
FMRI_DATA=drive_upload/cbp001 jupyter nbconvert --execute --to notebook notebooks/01_preprocessing.ipynb --output ../scratch.ipynb   # local test
```
