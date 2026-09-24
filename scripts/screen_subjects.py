"""Screen subjects the way the notebooks will show them. Uses derivatives/<sub>/<run>/bold_mni.nii.gz (2 mm).
Columns: motion (mean/max framewise displacement, mm), insula t-stat for the stimulus regressor,
NAc correlation with d(stim)/dt and |d(stim)/dt|, NAc event-locked response 5-12.5 s after onset and after offset (% signal change)."""
import numpy as np, nibabel as nib, glob, os, pandas as pd
from nilearn import image, masking
from nilearn.maskers import NiftiSpheresMasker
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.first_level.hemodynamic_models import spm_hrf
TR = 2.5; t = np.arange(240) * TR
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
hrf = spm_hrf(TR, oversampling=1, time_length=30)
nac_mask = image.math_img('img > 0', img=f'{ROOT}/subjects/accumbens_subdivionsLR.nii.gz')
D = make_first_level_design_matrix(t, drift_model='cosine', high_pass=0.01).values

def detrend(y): return y - D @ np.linalg.lstsq(D, y, rcond=None)[0]
def psc(y): return 100 * (y - y.mean()) / y.mean()
def tstat(y, reg):
    X = np.column_stack([reg, D]); b = np.linalg.lstsq(X, y, rcond=None)[0]; r = y - X @ b
    return b[0] / np.sqrt(r.var(ddof=X.shape[1]) * np.linalg.inv(X.T @ X)[0, 0])

rows = []
for d in sorted(glob.glob(f'{ROOT}/derivatives/*/task001_run00*/bold_mni.nii.gz')):
    sub, run = d.split('/')[-3], d.split('/')[-2]
    raw = f'{ROOT}/data/{sub}/bold/{run}'
    if not os.path.isdir(raw): raw = f'{ROOT}/data/OneDrive_1_9-23-2026/{sub}/bold/{run}'
    stim = np.loadtxt(f'{raw}/stimulus.txt'); rating = np.loadtxt(f'{raw}/behavior.txt')
    mp = np.loadtxt(os.path.dirname(d) + '/bold_mc.par'); fd = np.abs(np.diff(mp, axis=0)); fd[:, :3] *= 50; fd = fd.sum(1)
    img = image.smooth_img(d, 4)
    nac = detrend(psc(masking.apply_mask(img, image.resample_to_img(nac_mask, img, interpolation='nearest')).mean(1)))
    ins = NiftiSpheresMasker([(40, 8, -2), (-40, 8, -2)], radius=5).fit_transform(img); ins = np.array([detrend(psc(c)) for c in ins.T])
    box = (stim > 42).astype(float); reg = np.convolve(box, hrf)[:240]
    dstim = np.gradient(stim); deriv = np.convolve(dstim, hrf)[:240]; rect = np.convolve(np.abs(dstim), hrf)[:240]
    on = np.where(np.diff(box) == 1)[0] + 1; off = np.where(np.diff(box) == -1)[0] + 1
    lags = np.arange(2, 6)   # 5 to 12.5 s after the event
    locked = lambda ev: np.mean([nac[e + lags].mean() - nac[e - 2:e + 1].mean() for e in ev if e + lags[-1] < 240 and e >= 2])
    rows.append(dict(sub=sub, run=run[-1], meanFD=fd.mean(), maxFD=fd.max(),
                     insR_t=tstat(ins[0], reg), insL_t=tstat(ins[1], reg), insR_t_rating=tstat(ins[0], np.convolve(rating, hrf)[:240]),
                     nac_r_deriv=np.corrcoef(nac, deriv)[0, 1], nac_r_rect=np.corrcoef(nac, rect)[0, 1],
                     nac_onset=locked(on), nac_offset=locked(off)))
df = pd.DataFrame(rows); pd.set_option('display.width', 220); print(df.round(2).to_string())
df.to_csv(f'{ROOT}/derivatives/screening.csv', index=False)
