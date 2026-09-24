"""Build the three Colab notebooks from cell sources. Run: python3 scripts/build_notebooks.py
Cells beginning with '# %% md' become markdown cells; everything else is code.
"""
import nbformat as nbf, re, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'notebooks')
os.makedirs(OUT, exist_ok=True)

SETUP = '''#@title Setup: install packages and download the data (run this first, takes about a minute)
import os, sys
SUBJECT = 'SUBJECT_PLACEHOLDER'      # <-- change: 'cbp001', 'cbp006', 'cbp014' (chronic back pain) or 'healthy007'

DATA_URL = 'https://github.com/cmahlen/fmri-pain-class/releases/download/data-v1'   # one ~160 MB tarball per subject, one 10-minute run each

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    get_ipython().system('pip install -q --no-deps nilearn ipyniivue anywidget psygnal')   # only what Colab lacks; --no-deps keeps Colab's pandas/requests
    from google.colab import output; output.enable_custom_widget_manager()               # lets the clickable brain viewer render in Colab
    DATA = f'data/{SUBJECT}'
    if not os.path.exists(DATA):
        os.makedirs('data', exist_ok=True)
        get_ipython().system(f'curl -sL {DATA_URL}/{SUBJECT}.tar.gz | tar xz -C data')
else:
    DATA = os.path.join(os.environ.get('FMRI_DATA', 'drive_upload'), SUBJECT)

import numpy as np, nibabel as nib, matplotlib.pyplot as plt, pandas as pd, warnings
from nilearn import plotting, image, masking
from nilearn.datasets import load_mni152_template
warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 90
TR = 2.5                                   # seconds between volumes
t = np.arange(240) * TR                    # time axis in seconds
print('files:', sorted(os.listdir(DATA)))'''


LOCAL_SETUP = '''# Local version: reads the data straight from ../../drive_upload/<SUBJECT>/ (no Google Drive download)
import os, sys
SUBJECT = 'SUBJECT_PLACEHOLDER'      # <-- change: any folder name in drive_upload/
IN_COLAB = False
DATA = os.path.join(os.path.dirname(os.path.abspath('__file__')), '..', '..', 'drive_upload', SUBJECT)

import numpy as np, nibabel as nib, matplotlib.pyplot as plt, pandas as pd, warnings
from nilearn import plotting, image, masking
from nilearn.datasets import load_mni152_template
warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 90
TR = 2.5                                   # seconds between volumes
t = np.arange(240) * TR                    # time axis in seconds
print('files:', sorted(os.listdir(DATA)))'''


def build(name, cells, subject, local=False):
    if local:
        cells = [LOCAL_SETUP if c == SETUP else c for c in cells]
        name = os.path.join('local', name)
        os.makedirs(os.path.join(OUT, 'local'), exist_ok=True)
    cells = [c.replace('SUBJECT_PLACEHOLDER', subject) for c in cells]
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "colab": {"provenance": []}}
    for c in cells:
        c = c.strip('\n')
        if c.startswith('# %% md'):
            nb.cells.append(nbf.v4.new_markdown_cell(c.split('\n', 1)[1].strip()))
        else:
            nb.cells.append(nbf.v4.new_code_cell(c))
    path = os.path.join(OUT, name)
    nbf.write(nb, path)
    print('wrote', path)


# =====================================================================
# 1. PREPROCESSING
# =====================================================================
pre = [
'''# %% md
# Interactive 1: What does fMRI preprocessing do to the data?

One subject, one 10-minute run, painful heat applied to the lower back (Baliki et al., 2010, *Neuron*).

Each section shows the data **before** and **after** one preprocessing step. Lines marked `# <-- change` are the ones to play with.''',
SETUP,
'''# %% md
## 1. The raw data: a stack of 3D pictures taken every 2.5 seconds''',
'''raw = nib.load(f'{DATA}/bold_raw.nii.gz')
data = raw.get_fdata()
print('array shape (x, y, z, time):', data.shape)
print('voxel size (mm):', np.round(raw.header.get_zooms()[:3], 2))
print('TR (s):', TR, '   volumes:', data.shape[3], '   run length (min):', data.shape[3] * TR / 60)

plotting.plot_epi(image.index_img(raw, 0), title='volume 0 (the first 3D picture)', cmap='gray', draw_cross=False)
plotting.show()''',
'''# %% md
## 2. A voxel is a time series''',
'''x, y, z = 36, 30, 20      # <-- change: voxel position (x 0-63, y 0-63, z 0-35)

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].imshow(data[:, :, z, 0].T, cmap='gray', origin='lower');  ax[0].plot(x, y, 'r+', ms=18, mew=2); ax[0].set_title(f'axial slice z={z}')
ax[1].imshow(data[x, :, :, 0].T, cmap='gray', origin='lower', aspect=3 / 3.44); ax[1].plot(y, z, 'r+', ms=18, mew=2); ax[1].set_title(f'sagittal slice x={x}')
ax[2].plot(t, data[x, y, z, :], color='steelblue'); ax[2].set(xlabel='time (s)', ylabel='MRI signal (arbitrary units)', title='this voxel across the run')
plt.tight_layout(); plt.show()''',
'''# Same thing, but click around: click anywhere in the viewer and the plot underneath shows that voxel's time series
from ipyniivue import NiiVue, SliceType
from ipywidgets import Output
from IPython.display import display, clear_output

viewer_img = image.mean_img(raw)
viewer_img.set_sform(raw.affine, code=1); viewer_img.set_qform(raw.affine, code=1)   # raw file has no stored orientation; write one so the viewer and numpy agree
viewer_img.to_filename('mean_raw.nii.gz')
nv = NiiVue(slice_type=SliceType.MULTIPLANAR, height=420)
nv.load_volumes([{'path': 'mean_raw.nii.gz', 'colormap': 'gray'}])
out = Output()

@nv.on_location_change
def show_voxel(loc):
    i, j, k = np.round(np.linalg.inv(viewer_img.affine) @ [*loc['mm'][:3], 1])[:3].astype(int)
    if not (0 <= i < data.shape[0] and 0 <= j < data.shape[1] and 0 <= k < data.shape[2]):
        return                                               # clicked outside the image
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, data[i, j, k], color='steelblue'); ax.set(title=f'voxel ({i}, {j}, {k})', xlabel='time (s)')
    plt.close(fig)
    with out:
        clear_output(wait=True); display(fig)

display(nv, out)''',
'''# %% md
## 3. The whole brain across time: does it stay still?''',
'''volumes = [0, 60, 120, 180, 239]      # <-- change: which volumes to look at
z = 20

brain = data.mean(-1) > 0.5 * data.mean()                                  # rough brain mask: voxels brighter than half the average
psc = np.where(brain[..., None], 100 * (data - data.mean(-1, keepdims=True)) / (data.mean(-1, keepdims=True) + 1), np.nan)   # percent signal change, background blanked

fig, ax = plt.subplots(2, len(volumes), figsize=(3.6 * len(volumes), 7))
for i, v in enumerate(volumes):
    ax[0, i].imshow(data[:, :, z, v].T, cmap='gray', origin='lower'); ax[0, i].set_title(f'volume {v}  (t = {v * TR:.0f} s)')
    im = ax[1, i].imshow((psc[:, :, z, v] - psc[:, :, z, 0]).T, cmap='RdBu_r', vmin=-20, vmax=20, origin='lower'); ax[1, i].set_title(f'vol {v} - vol 0  (% change)')
for a in ax.ravel(): a.axis('off')
fig.colorbar(im, ax=ax.ravel().tolist(), label='% signal change', shrink=0.5, pad=0.02); plt.show()''',
'''# %% md
## 4. Head motion correction
Every volume is rigidly shifted and rotated to line up with the first one. The estimated movement is saved as 6 numbers per volume.''',
'''motion = np.loadtxt(f'{DATA}/motion_params.txt')      # columns: 3 rotations (radians), 3 translations (mm)

fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
ax[0].plot(t, np.degrees(motion[:, :3])); ax[0].legend(['pitch', 'roll', 'yaw']); ax[0].set_ylabel('rotation (degrees)')
ax[1].plot(t, motion[:, 3:]);             ax[1].legend(['x', 'y', 'z']);          ax[1].set_ylabel('translation (mm)'); ax[1].set_xlabel('time (s)')
plt.show()

worst = np.abs(motion - motion[0]).sum(1).argmax()
print('volume that moved the most relative to volume 0:', worst, f'(t = {worst * TR:.0f} s)')''',
'''mc = nib.load(f'{DATA}/bold_mc.nii.gz').get_fdata()   # the same run after motion correction
v = worst                                            # <-- change: any volume number

before_diff = np.abs(data[..., v] - data[..., 0]).sum((0, 1)); after_diff = np.abs(mc[..., v] - mc[..., 0]).sum((0, 1))
zbest = 5 + (before_diff - after_diff)[5:30].argmax()                     # slice where motion correction helped the most
fig, ax = plt.subplots(2, 3, figsize=(15, 9))
for row, (label, d) in enumerate([('BEFORE motion correction', data), ('AFTER motion correction', mc)]):
    diff = np.where(brain, 100 * (d[..., v] - d[..., 0]) / (d.mean(-1) + 1), np.nan)
    ax[row, 0].imshow(d[:, :, zbest, v].T, cmap='gray', origin='lower');            ax[row, 0].set_title(f'{label}\\nvolume {v}, slice z={zbest}')
    ax[row, 1].imshow(diff[:, :, zbest].T, cmap='RdBu_r', vmin=-20, vmax=20, origin='lower'); ax[row, 1].set_title(f'volume {v} minus volume 0 (% signal change)')
    im = ax[row, 2].imshow(diff[32, :, :].T, cmap='RdBu_r', vmin=-20, vmax=20, origin='lower', aspect=3 / 3.44); ax[row, 2].set_title('same difference, sagittal view')
for a in ax.ravel(): a.axis('off')
fig.colorbar(im, ax=ax[:, 2], label='% signal change', shrink=0.6); plt.show()''',
'''# A voxel at the edge of the brain: motion pretends to be brain activity
improvement = np.where(brain, data.std(-1) - mc.std(-1), 0)
x, y, z = np.unravel_index(improvement.argmax(), improvement.shape)   # <-- change: or pick your own x, y, z

def show_voxel_location(x, y, z, ax):
    ax[0].imshow(mc[:, :, z, 0].T, cmap='gray', origin='lower'); ax[0].plot(x, y, 'r+', ms=18, mew=2); ax[0].set_title(f'axial z={z}'); ax[0].axis('off')
    ax[1].imshow(mc[x, :, :, 0].T, cmap='gray', origin='lower', aspect=3 / 3.44); ax[1].plot(y, z, 'r+', ms=18, mew=2); ax[1].set_title(f'sagittal x={x}'); ax[1].axis('off')

fig, ax = plt.subplots(1, 2, figsize=(9, 4)); show_voxel_location(x, y, z, ax); plt.tight_layout(); plt.show()

fig, ax = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
ax[0].plot(t, data[x, y, z], color='firebrick');   ax[0].set_title(f'voxel ({x}, {y}, {z}) BEFORE motion correction')
ax[1].plot(t, mc[x, y, z], color='seagreen');      ax[1].set_title('AFTER motion correction')
ax[2].plot(t, np.degrees(motion[:, 0]), color='gray'); ax[2].set_title('estimated head rotation (degrees)'); ax[2].set_xlabel('time (s)')
plt.tight_layout(); plt.show()''',
'''# %% md
## 5. Temporal filtering
Slow drifts (scanner warming up, slow physiology) are removed with a high-pass filter.''',
'''from nilearn.signal import clean
from scipy.signal import welch

HIGH_PASS = 0.01     # Hz  <-- change: try 0.005, 0.02, 0.05
LOW_PASS  = None     # Hz  <-- change: try 0.1 (removes fast noise too)

signal = mc[brain].mean(0)                 # average of all brain voxels = "global signal"

def temporal_filter(x):                    # band-pass filter, then put the original mean back so before and after overlay
    y = clean(x[:, None], t_r=TR, high_pass=HIGH_PASS, low_pass=LOW_PASS, detrend=False, standardize=False)[:, 0]
    return y - y.mean() + x.mean()

filtered = temporal_filter(signal)

trend = np.polyval(np.polyfit(t, signal, 1), t)      # straight line fitted to the "before" signal: the slow drift

fig, ax = plt.subplots(1, 2, figsize=(16, 4))
ax[0].plot(t, signal, color='firebrick', label='before'); ax[0].plot(t, trend, 'k--', lw=2, label='linear trend of before'); ax[0].plot(t, filtered, color='seagreen', label='after'); ax[0].legend(); ax[0].set(xlabel='time (s)', title='global mean signal')
for s, c, l in [(signal, 'firebrick', 'before'), (filtered, 'seagreen', 'after')]:
    f, p = welch(s - s.mean(), fs=1 / TR, nperseg=120); ax[1].semilogy(f, p, color=c, label=l)
ax[1].axvline(HIGH_PASS, ls='--', color='k'); ax[1].legend(); ax[1].set(xlabel='frequency (Hz)', ylabel='power', title='power spectrum')
plt.show()''',
'''x, y, z = 36, 30, 20      # <-- change: a single voxel

signal = mc[x, y, z]
filtered = temporal_filter(signal)
fig, ax = plt.subplots(1, 3, figsize=(17, 4), width_ratios=[1, 1, 3])
show_voxel_location(x, y, z, ax)
trend = np.polyval(np.polyfit(t, signal, 1), t)
ax[2].plot(t, signal, color='firebrick', label='before'); ax[2].plot(t, trend, 'k--', lw=2, label='linear trend of before'); ax[2].plot(t, filtered, color='seagreen', label='after')
ax[2].legend(); ax[2].set(xlabel='time (s)', title=f'voxel ({x}, {y}, {z})')
plt.tight_layout(); plt.show()''',
'''# %% md
## 6. Spatial smoothing
Each voxel is replaced by a weighted average of its neighbours (Gaussian kernel, width given as FWHM in mm).''',
'''FWHM_LIST = [0, 4, 8, 12]      # <-- change: smoothing kernel sizes in mm

mc_img = nib.load(f'{DATA}/bold_mc.nii.gz')
vol = image.index_img(mc_img, 100)
fig, ax = plt.subplots(1, len(FWHM_LIST), figsize=(4.5 * len(FWHM_LIST), 4.5))
for a, fwhm in zip(ax, FWHM_LIST):
    sm = image.smooth_img(vol, fwhm) if fwhm else vol
    a.imshow(sm.get_fdata()[:, :, 20].T, cmap='gray', origin='lower'); a.set_title(f'FWHM = {fwhm} mm'); a.axis('off')
plt.tight_layout(); plt.show()''',
'''FWHM = 6                    # <-- change
x, y, z = 36, 30, 20        # <-- change

smoothed = image.smooth_img(mc_img, FWHM).get_fdata()
tsnr_before = mc.mean(-1) / (mc.std(-1) + 1e-6) * brain
tsnr_after = smoothed.mean(-1) / (smoothed.std(-1) + 1e-6) * brain

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].imshow(tsnr_before[:, :, z].T, cmap='hot', vmin=0, vmax=150, origin='lower'); ax[0].set_title('signal-to-noise per voxel, BEFORE smoothing'); ax[0].axis('off')
im = ax[1].imshow(tsnr_after[:, :, z].T,  cmap='hot', vmin=0, vmax=150, origin='lower'); ax[1].set_title(f'AFTER smoothing (FWHM = {FWHM} mm)'); ax[1].axis('off')
fig.colorbar(im, ax=ax, label='temporal SNR (mean / std)', shrink=0.8); plt.show()

fig, ax = plt.subplots(1, 3, figsize=(17, 4), width_ratios=[1, 1, 3])
show_voxel_location(x, y, z, ax)
ax[2].plot(t, mc[x, y, z], color='firebrick', label='before'); ax[2].plot(t, smoothed[x, y, z], color='seagreen', label='after'); ax[2].legend(); ax[2].set(xlabel='time (s)', title=f'voxel ({x}, {y}, {z})')
plt.tight_layout(); plt.show()''',
'''# %% md
## 7. Registration
The functional images are aligned to the subject's own T1 image, and then to a standard template (MNI152), so that a coordinate like (16, 10, -8) means the same place in everyone.''',
'''mni = load_mni152_template(resolution=2)
t1 = nib.load(f'{DATA}/t1_brain.nii.gz')
mean_raw = image.mean_img(raw)
mean_mni = image.mean_img(nib.load(f'{DATA}/bold_mni.nii.gz'))

plotting.plot_anat(mean_raw, title='functional image, scanner space (native)', cmap='gray', draw_cross=False)
plotting.plot_anat(t1, title='T1 anatomical image, scanner space (native)', draw_cross=False)
plotting.plot_anat(mni, title='MNI152 template', draw_cross=False)
plotting.show()''',
'''CUT = (10, 12, -8)      # <-- change: MNI coordinate to look at (this one is the right nucleus accumbens)

d = plotting.plot_anat(mean_raw, cut_coords=CUT, title='BEFORE: native functional image with template outline', cmap='gray'); d.add_edges(mni)
d = plotting.plot_anat(mean_mni, cut_coords=CUT, title='AFTER: registered functional image with template outline', cmap='gray'); d.add_edges(mni)
plotting.show()''',
'''# %% md
## 8. All steps on one voxel''',
'''X_MNI, Y_MNI, Z_MNI = 40, 8, -2      # <-- change: MNI coordinate. (40, 8, -2) is the right anterior insula, a region that responds to pain

from nilearn.maskers import NiftiSpheresMasker
mni_img = nib.load(f'{DATA}/bold_mni.nii.gz')
raw_mni_img = nib.load(f'{DATA}/bold_raw_mni.nii.gz')      # the raw run pushed through the same registration, no motion correction
d = plotting.plot_anat(mean_mni, cut_coords=(X_MNI, Y_MNI, Z_MNI), title='the voxel we are following', cmap='gray')
d.add_markers([(X_MNI, Y_MNI, Z_MNI)], marker_color='red', marker_size=120)
plotting.show()
stim = np.loadtxt(f'{DATA}/stimulus.txt')
sphere = NiftiSpheresMasker([(X_MNI, Y_MNI, Z_MNI)], radius=0)
uncorrected_ts = sphere.fit_transform(raw_mni_img)[:, 0]
raw_ts = sphere.fit_transform(mni_img)[:, 0]
filt_ts = temporal_filter(raw_ts)
smooth_ts = NiftiSpheresMasker([(X_MNI, Y_MNI, Z_MNI)], radius=0).fit_transform(image.smooth_img(mni_img, 6))[:, 0]
smooth_filt_ts = temporal_filter(smooth_ts)

fig, ax = plt.subplots(5, 1, figsize=(13, 13), sharex=True)
ax[0].plot(t, uncorrected_ts, color='firebrick');  ax[0].set_title('raw (registered only, no motion correction)')
ax[1].plot(t, raw_ts, color='darkorange');         ax[1].set_title('motion corrected')
ax[2].plot(t, filt_ts, color='steelblue');    ax[2].set_title('+ high-pass filtered')
ax[3].plot(t, smooth_filt_ts, color='seagreen'); ax[3].set_title('+ smoothed (6 mm)')
ax[4].plot(t, stim, color='firebrick');       ax[4].set_title('heat stimulus (C)'); ax[4].set_xlabel('time (s)')
plt.tight_layout(); plt.show()''',
'''# Same traces on top of each other, as percent signal change. Tick the boxes to choose which ones to show.
from ipywidgets import Checkbox, HBox, interactive_output

traces = {'raw': uncorrected_ts, 'motion corrected': raw_ts, 'high-pass filtered': filt_ts, 'smoothed': smooth_filt_ts}
colors = {'raw': 'firebrick', 'motion corrected': 'darkorange', 'high-pass filtered': 'steelblue', 'smoothed': 'seagreen'}
boxes = {name: Checkbox(value=(name in ['raw', 'smoothed']), description=name) for name in traces}
boxes['heat stimulus'] = Checkbox(value=True, description='heat stimulus')

def overlay(**show):
    fig, ax = plt.subplots(figsize=(14, 5))
    for name, ts in traces.items():
        if show[name]:
            ax.plot(t, 100 * (ts - ts.mean()) / raw_ts.mean(), color=colors[name], alpha=0.6, label=name)   # all relative to the same baseline
    ax.set(xlabel='time (s)', ylabel='% signal change', title=f'MNI ({X_MNI}, {Y_MNI}, {Z_MNI})'); ax.legend(loc='upper left')
    if show['heat stimulus']:
        ax2 = ax.twinx(); ax2.plot(t, stim, color='k', alpha=0.25, lw=3); ax2.set_ylabel('temperature (C)')
    plt.show()

display(HBox(list(boxes.values())), interactive_output(overlay, boxes))''',
]

# =====================================================================
# 2. GLM
# =====================================================================
glm = [
'''# %% md
# Interactive 2: Finding pain in the brain with the general linear model

Same subject, same run. Now we ask: *which voxels go up and down with the heat stimulus?*''',
SETUP,
'''# %% md
## 1. What happened during the scan''',
'''bold = nib.load(f'{DATA}/bold_mni.nii.gz')            # preprocessed run (motion corrected, registered to MNI, 3 mm voxels)
stim = np.loadtxt(f'{DATA}/stimulus.txt')               # probe temperature (C) at each volume
rating = np.loadtxt(f'{DATA}/rating.txt')               # subject's pain rating (0-100) at each volume
motion = np.loadtxt(f'{DATA}/motion_params.txt')

fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
ax[0].plot(t, stim, color='firebrick');  ax[0].set_ylabel('temperature (C)')
ax[1].plot(t, rating, color='k');        ax[1].set_ylabel('pain rating (0-100)'); ax[1].set_xlabel('time (s)')
plt.show()''',
'''# %% md
## 2. From stimulus to expected BOLD signal: the hemodynamic response function''',
'''from nilearn.glm.first_level.hemodynamic_models import spm_hrf

STIM_THRESHOLD = 42      # C  <-- change: temperature above which we call it "stimulus on"

hrf_fine = spm_hrf(0.1, oversampling=1, time_length=30)
hrf = spm_hrf(TR, oversampling=1, time_length=30)
boxcar = (stim > STIM_THRESHOLD).astype(float)
expected = np.convolve(boxcar, hrf)[:240]

fig, ax = plt.subplots(3, 1, figsize=(13, 8))
ax[0].plot(np.arange(len(hrf_fine)) * 0.1, hrf_fine, color='purple'); ax[0].set(title='hemodynamic response to a brief event', xlabel='seconds after event')
ax[1].plot(t, boxcar, color='firebrick'); ax[1].set(title='stimulus on / off', ylim=(-0.1, 1.2))
ax[2].plot(t, expected, color='purple');  ax[2].set(title='stimulus convolved with the HRF = expected BOLD signal', xlabel='time (s)')
plt.tight_layout(); plt.show()''',
'''# %% md
## 3. The GLM in one voxel: how well does the expected signal explain the measured one?''',
'''from nilearn.maskers import NiftiSpheresMasker
from nilearn.glm.first_level import make_first_level_design_matrix

COORDS = (40, 8, -2)      # <-- change: MNI coordinate. (40, 8, -2) right insula; (16, 10, -8) right accumbens; (-38, -22, 56) left motor cortex

def design(regressors, high_pass=0.01):
    return make_first_level_design_matrix(t, add_regs=pd.DataFrame(regressors, index=t), drift_model='cosine', high_pass=high_pass)

dm = design({'pain': expected})
y = NiftiSpheresMasker([COORDS], radius=4).fit_transform(bold)[:, 0]
y = 100 * (y - y.mean()) / y.mean()                                  # percent signal change

X = dm.values
beta = np.linalg.lstsq(X, y, rcond=None)[0]
fit = X @ beta
resid = y - fit
i = list(dm.columns).index('pain')
se = np.sqrt(resid.var(ddof=X.shape[1]) * np.linalg.inv(X.T @ X)[i, i])
print(f'beta for pain = {beta[i]:.3f} % signal change     t = {beta[i] / se:.2f}')

fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
ax[0].plot(t, y, color='gray', label='measured'); ax[0].plot(t, fit, color='purple', lw=2, label='GLM fit'); ax[0].legend(); ax[0].set_title(f'voxel at MNI {COORDS}')
ax[1].plot(t, resid, color='gray'); ax[1].set_title('residual (what the model does not explain)'); ax[1].set_xlabel('time (s)')
plt.tight_layout(); plt.show()''',
'''# %% md
## 4. The design matrix: the same model, written as columns''',
'''INCLUDE_MOTION = False      # <-- change: add the 6 head-motion parameters as nuisance regressors

regs = {'pain': expected}
if INCLUDE_MOTION:
    regs.update({f'motion{i}': motion[:, i] for i in range(6)})
dm = design(regs)
plotting.plot_design_matrix(dm)
plotting.show()''',
'''# %% md
## 5. The GLM in every voxel''',
'''from nilearn.glm.first_level import FirstLevelModel

SMOOTHING = 6       # mm  <-- change: 0, 4, 6, 8
THRESHOLD = 3.1     # z   <-- change

model = FirstLevelModel(t_r=TR, smoothing_fwhm=SMOOTHING, minimize_memory=False).fit(bold, design_matrices=dm)
zmap = model.compute_contrast('pain')

plotting.plot_stat_map(zmap, threshold=THRESHOLD, cut_coords=COORDS, title=f'pain > rest, z > {THRESHOLD}, smoothing {SMOOTHING} mm')
plotting.plot_glass_brain(zmap, threshold=THRESHOLD, colorbar=True, plot_abs=False, display_mode='lyrz')
plotting.show()''',
'''# Click on the map: the plot underneath shows that voxel's data and its GLM fit
from ipyniivue import NiiVue, SliceType
from ipywidgets import Output
from IPython.display import display, clear_output

mni = load_mni152_template(resolution=2); mni.to_filename('mni.nii.gz'); zmap.to_filename('zmap.nii.gz')
bold_smooth = image.smooth_img(bold, SMOOTHING).get_fdata()
nv = NiiVue(slice_type=SliceType.MULTIPLANAR, height=450)
nv.load_volumes([{'path': 'mni.nii.gz', 'colormap': 'gray'},
                 {'path': 'zmap.nii.gz', 'colormap': 'warm', 'opacity': 0.8, 'cal_min': THRESHOLD, 'cal_max': 8}])
nv.opts.is_colorbar = True
out = Output()

zmap_data = zmap.get_fdata()

@nv.on_location_change
def show_fit(loc):
    i, j, k = np.round(np.linalg.inv(bold.affine) @ [*loc['mm'][:3], 1])[:3].astype(int)
    zi, zj, zk = np.round(np.linalg.inv(zmap.affine) @ [*loc['mm'][:3], 1])[:3].astype(int)
    y = bold_smooth[i, j, k]; y = 100 * (y - y.mean()) / y.mean()
    b = np.linalg.lstsq(dm.values, y, rcond=None)[0]
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, y, color='gray', label='measured'); ax.plot(t, dm.values @ b, color='purple', lw=2, label='GLM fit')
    ax.set(title=f'MNI {tuple(np.round(loc["mm"][:3]).astype(int))}    z = {zmap_data[zi, zj, zk]:.2f}', xlabel='time (s)'); ax.legend(loc='upper right')
    plt.close(fig)
    with out:
        clear_output(wait=True); display(fig)

display(nv, out)''',
'''# The same map as a self-contained interactive viewer (no clicking into Python, but works everywhere)
plotting.view_img(zmap, threshold=THRESHOLD, cut_coords=COORDS, title='pain > rest')''',
'''# %% md
## 6. Multiple comparisons: how many voxels did we just test?''',
'''from nilearn.glm import threshold_stats_img

n_voxels = int(model.masker_.mask_img_.get_fdata().sum())
print('voxels tested:', n_voxels, '   expected false positives at p < 0.05 if nothing is happening:', round(0.05 * n_voxels))

for label, kw in [('uncorrected p < 0.05', dict(alpha=0.05, height_control='fpr')),
                  ('uncorrected p < 0.001', dict(alpha=0.001, height_control='fpr')),
                  ('FDR corrected q < 0.05', dict(alpha=0.05, height_control='fdr')),
                  ('Bonferroni corrected p < 0.05', dict(alpha=0.05, height_control='bonferroni'))]:
    thr_map, thr = threshold_stats_img(zmap, **kw)
    plotting.plot_glass_brain(thr_map, colorbar=True, plot_abs=False, display_mode='lyrz', title=f'{label}  (z > {thr:.2f})')
plotting.show()''',
'''# %% md
## 7. Two models of the same data: stimulus temperature vs perceived pain''',
'''rating_expected = np.convolve(rating - rating.min(), hrf)[:240]

for name, reg in [('stimulus', expected), ('rating', rating_expected)]:
    m = FirstLevelModel(t_r=TR, smoothing_fwhm=SMOOTHING).fit(bold, design_matrices=design({name: reg}))
    plotting.plot_stat_map(m.compute_contrast(name), threshold=THRESHOLD, cut_coords=COORDS, title=f'model: {name}')
plotting.show()''',
'''# %% md
## 8. The nucleus accumbens: responding to change, not to heat
Baliki et al. found the accumbens responds when the stimulus *starts* and when it *ends*. In healthy subjects both responses are positive (BOLD follows |d stim/dt|). In chronic back pain the offset response flips sign (BOLD follows d stim/dt).''',
'''nac_mask = image.math_img('img > 0', img=f'{DATA}/nac_mask_mni.nii.gz')
nac = masking.apply_mask(image.smooth_img(bold, 4), image.resample_to_img(nac_mask, bold, interpolation='nearest')).mean(1)
nac = 100 * (nac - nac.mean()) / nac.mean()
D = design({}).values                                   # drift-only design matrix
nac = nac - D @ np.linalg.lstsq(D, nac, rcond=None)[0]   # remove slow drift

dstim = np.gradient(stim)
deriv = np.convolve(dstim, hrf)[:240]                 # d stim / dt   (positive at onset, negative at offset)
rect  = np.convolve(np.abs(dstim), hrf)[:240]         # |d stim / dt| (positive at onset AND offset)

fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
ax[0].plot(t, nac, color='k');            ax[0].set_title('nucleus accumbens BOLD (% signal change)')
ax[1].plot(t, deriv, color='firebrick');  ax[1].set_title(f'd stim/dt convolved with HRF      correlation with NAc = {np.corrcoef(nac, deriv)[0, 1]:.2f}')
ax[2].plot(t, rect, color='seagreen');    ax[2].set_title(f'|d stim/dt| convolved with HRF     correlation with NAc = {np.corrcoef(nac, rect)[0, 1]:.2f}'); ax[2].set_xlabel('time (s)')
plt.tight_layout(); plt.show()''',
'''# Average the accumbens signal around every stimulus onset and every stimulus offset
WINDOW = (-10, 30)       # seconds  <-- change

onsets  = np.where(np.diff(boxcar) == 1)[0] + 1
offsets = np.where(np.diff(boxcar) == -1)[0] + 1
lags = np.arange(int(WINDOW[0] / TR), int(WINDOW[1] / TR))

def locked(events, series):
    return np.array([series[e + lags] for e in events if e + lags[0] >= 0 and e + lags[-1] < 240])

fig, ax = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
for a, ev, name in [(ax[0], onsets, 'stimulus ONSET'), (ax[1], offsets, 'stimulus OFFSET')]:
    seg = locked(ev, nac)
    a.plot(lags * TR, seg.T, color='lightgray')
    a.plot(lags * TR, seg.mean(0), color='k', lw=3, label=f'mean of {len(seg)} events')
    a.axvline(0, ls='--', color='firebrick'); a.axhline(0, color='gray', lw=0.5); a.legend(); a.set(title=f'NAc around {name}', xlabel='seconds from event')
plt.show()''',
]

# =====================================================================
# 3. CONNECTIVITY
# =====================================================================
fc = [
'''# %% md
# Interactive 3: Functional connectivity

Same subject, same run. Now we ignore the task and ask: *which parts of the brain rise and fall together?*''',
SETUP,
'''# %% md
## 1. Prepare the data: band-pass filter, remove head motion, z-score every voxel''',
'''from nilearn.maskers import NiftiMasker, NiftiSpheresMasker

bold = nib.load(f'{DATA}/bold_mni.nii.gz')
motion = np.loadtxt(f'{DATA}/motion_params.txt')
stim = np.loadtxt(f'{DATA}/stimulus.txt')

HIGH_PASS, LOW_PASS = 0.009, 0.08      # Hz  <-- change
SMOOTHING = 6                          # mm  <-- change
REMOVE_GLOBAL_SIGNAL = True            # <-- change: also regress out the average signal of the whole brain

brain_mask = masking.compute_epi_mask(bold)
global_signal = masking.apply_mask(bold, brain_mask).mean(1)
confounds = np.column_stack([motion, global_signal]) if REMOVE_GLOBAL_SIGNAL else motion
clean = image.clean_img(image.smooth_img(bold, SMOOTHING), t_r=TR, confounds=confounds, high_pass=HIGH_PASS, low_pass=LOW_PASS, detrend=True, standardize='zscore_sample', mask_img=brain_mask)
masker = NiftiMasker(mask_img=brain_mask).fit()
X = masker.transform(clean)            # matrix: 240 time points x all brain voxels
print('time points x voxels:', X.shape)''',
'''# %% md
## 2. Seed-based connectivity: correlate one region with every voxel''',
'''SEED = (16, 10, -8)      # MNI  <-- change: (16, 10, -8) right accumbens; (0, 52, -14) medial prefrontal; (40, 8, -2) right insula; (-38, -22, 56) left motor cortex
RADIUS = 5               # mm
R_THRESHOLD = 0.4        # <-- change

seed = NiftiSpheresMasker([SEED], radius=RADIUS).fit_transform(clean)[:, 0]
seed = (seed - seed.mean()) / seed.std(ddof=1)
r = X.T @ seed / (len(seed) - 1)                     # correlation of the seed with every voxel
r_img = masker.inverse_transform(r)

plt.figure(figsize=(13, 3)); plt.plot(t, seed, color='k'); plt.title(f'seed time series at MNI {SEED}'); plt.xlabel('time (s)'); plt.show()
plotting.plot_stat_map(r_img, threshold=R_THRESHOLD, cut_coords=SEED, title=f'correlation with seed {SEED}', vmax=1)
plotting.plot_glass_brain(r_img, threshold=R_THRESHOLD, colorbar=True, plot_abs=False, display_mode='lyrz', vmax=1)
plotting.show()''',
'''# What a correlation of 0.7, 0.0 and -0.4 look like: the seed (black) against one voxel (colored) at each strength
TARGET_R = [0.7, 0.0, -0.4]                            # <-- change
target_colors = ['seagreen', 'orange', 'royalblue']

voxel_ijk = np.argwhere(brain_mask.get_fdata() > 0)   # voxel indices, same order as the columns of X
targets = []
for target_r, color in zip(TARGET_R, target_colors):
    j = np.abs(r - target_r).argmin()
    xyz = tuple(np.round(image.coord_transform(*voxel_ijk[j], brain_mask.affine)).astype(int))
    targets.append((j, xyz, color))

d = plotting.plot_glass_brain(None, display_mode='lyrz', title='seed (black) and the three example voxels')
d.add_markers([SEED], marker_color='k', marker_size=120)
for j, xyz, color in targets:
    d.add_markers([xyz], marker_color=color, marker_size=120)
plotting.show()

for j, xyz, color in targets:
    plt.figure(figsize=(13, 2.5)); plt.plot(t, seed, color='k', label='seed'); plt.plot(t, X[:, j], color=color, label=f'voxel at MNI {xyz}')
    plt.title(f'r = {r[j]:.2f}'); plt.legend(loc='upper right'); plt.show()''',
'''# %% md
## 3. A correlation matrix between regions''',
'''ROIS = {                       # <-- change: add or remove regions (name: MNI coordinate)
    'NAc R': (16, 10, -8),   'NAc L': (-16, 10, -8),
    'insula R': (40, 8, -2), 'insula L': (-40, 8, -2),
    'S2 R': (58, -22, 18),   'S2 L': (-58, -22, 18),
    'thalamus R': (10, -18, 6), 'thalamus L': (-10, -18, 6),
    'ACC': (2, 20, 32),      'mPFC': (0, 52, -14),
    'PCC': (0, -52, 26),     'motor L': (-38, -22, 56),
}
roi_ts = NiftiSpheresMasker(list(ROIS.values()), radius=RADIUS).fit_transform(clean)
corr = np.corrcoef(roi_ts.T)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(ROIS))); ax.set_xticklabels(ROIS, rotation=90); ax.set_yticks(range(len(ROIS))); ax.set_yticklabels(ROIS)
plt.colorbar(im, label='correlation'); plt.show()

plotting.plot_connectome(corr, list(ROIS.values()), edge_threshold=0.4, node_size=60, title='edges with |r| > 0.4')
plotting.show()''',
'''# %% md
## 4. Degree: how many strong connections does each voxel have?''',
'''DEGREE_THRESHOLD = 0.7     # <-- change: r above which we call two voxels "connected"

n = X.shape[0]
degree = np.zeros(X.shape[1])
for start in range(0, X.shape[1], 2000):              # correlate every voxel with every other voxel, in chunks
    chunk = X[:, start:start + 2000].T @ X / (n - 1)
    degree[start:start + 2000] = (chunk >= DEGREE_THRESHOLD).sum(1) - 1

plotting.plot_stat_map(masker.inverse_transform(degree), threshold=np.percentile(degree, 80), cmap='hot', title=f'degree (number of voxels with r >= {DEGREE_THRESHOLD})', display_mode='z', cut_coords=6)
plotting.show()''',
'''# %% md
## 5. Why preprocessing matters for connectivity''',
'''SEED = (0, 52, -14)      # <-- change

def seed_map(img, label):
    m = NiftiMasker(mask_img=brain_mask, standardize='zscore_sample').fit()
    Xi = m.transform(img)
    s = NiftiSpheresMasker([SEED], radius=RADIUS, standardize='zscore_sample').fit_transform(img)[:, 0]
    plotting.plot_stat_map(m.inverse_transform(Xi.T @ s / (len(s) - 1)), threshold=R_THRESHOLD, cut_coords=SEED, title=label, vmax=1)

seed_map(bold, 'raw data, no filtering, no motion regression')
seed_map(image.clean_img(bold, t_r=TR, high_pass=HIGH_PASS, low_pass=LOW_PASS, detrend=True, mask_img=brain_mask), 'band-pass filtered only')
seed_map(image.clean_img(bold, t_r=TR, confounds=motion, high_pass=HIGH_PASS, low_pass=LOW_PASS, detrend=True, mask_img=brain_mask), 'filtered + motion regressed')
seed_map(clean, 'filtered + motion regressed' + (' + global signal regressed' if REMOVE_GLOBAL_SIGNAL else '') + ' + smoothed')
plotting.show()''',
]

build('01_preprocessing.ipynb', pre, 'cbp001')    # large head motion: best for the motion-correction demo
build('02_glm.ipynb', glm, 'cbp014')              # strong pain map, highest NAc correlation with d(stim)/dt (run 1)
build('03_connectivity.ipynb', fc, 'cbp006')
for name, cells, sub in [('01_preprocessing.ipynb', pre, 'cbp001'), ('02_glm.ipynb', glm, 'cbp014'), ('03_connectivity.ipynb', fc, 'cbp006')]:
    build(name, cells, sub, local=True)      # notebooks/local/: same notebooks reading from drive_upload/ on this machine
