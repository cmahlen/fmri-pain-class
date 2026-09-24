#!/bin/bash
# Build the Google Drive upload folder for one subject/run: drive_upload/<sub>/
# usage: scripts/make_bundle.sh cbp001 task001_run001
set -e
export FSLOUTPUTTYPE=NIFTI_GZ
sub=$1; run=${2:-task001_run001}
ROOT=/Users/cmahlen/random_code_projects/paul_mri_class
D=$ROOT/derivatives/$sub/$run; O=$ROOT/drive_upload/$sub; mkdir -p $O
MNI=$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz
RAW=$ROOT/data/$sub/bold/$run; [ -d $RAW ] || RAW=$ROOT/data/OneDrive_1_9-23-2026/$sub/bold/$run
if [ -f $D/bold_mc_brain.nii.gz ]; then      # full pipeline (preprocess_fsl.sh): 3 mm MNI-space BOLD stored as int16 to keep the download small
  flirt -in $D/bold_mc_brain -ref $MNI -applyisoxfm 3 -init $D/epi2mni.mat -out $O/bold_mni -interp trilinear -datatype short
  fslmaths $D/bold_mc $O/bold_mc -odt short
else                                          # light pipeline (preprocess_light.sh) already wrote these
  cp $D/bold_mni.nii.gz $O/bold_mni.nii.gz
  cp $D/bold_mc.nii.gz $O/bold_mc.nii.gz
fi
cp $RAW/bold.nii.gz $O/bold_raw.nii.gz
# raw (not motion corrected) run pushed through the same registration, so raw and preprocessed can be compared at one MNI coordinate
flirt -in $RAW/bold.nii.gz -ref $MNI -applyisoxfm 3 -init $D/epi2mni.mat -out $O/bold_raw_mni -interp trilinear -datatype short
cp $D/bold_mc.par $O/motion_params.txt
cp $RAW/stimulus.txt $O/stimulus.txt
cp $RAW/behavior.txt $O/rating.txt
python3 - <<PY
import nibabel as nib, numpy as np
im = nib.load('$ROOT/derivatives/$sub/t1_brain.nii.gz'); a = im.affine.copy()
a[:3, 3] = -a[:3, :3] @ (np.array(im.shape) / 2)        # put the origin at the image centre (scanner offset was in the thousands of mm)
nib.save(nib.Nifti1Image(np.asarray(im.dataobj), a), '$O/t1_brain.nii.gz')
PY
cp $ROOT/derivatives/$sub/t1_mni.nii.gz $O/t1_mni.nii.gz
cp $ROOT/subjects/accumbens_subdivionsLR.nii.gz $O/nac_mask_mni.nii.gz
du -sh $O/*
