#!/bin/bash
# Local FSL preprocessing for all 6 subjects. Outputs go to derivatives/<sub>/<run>/
# Steps: mcflirt (motion correction + params), brain extraction, EPI->T1 (BBR-free 6dof), T1->MNI (12dof + fnirt skipped for speed), apply to BOLD.
set -e
export FSLOUTPUTTYPE=NIFTI_GZ
ROOT=/Users/cmahlen/random_code_projects/paul_mri_class
MNI=$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz
SUBS=${@:-"cbp001 cbp002 cbp003 healthy001 healthy003 healthy004"}   # healthy002 has no T1 in the download
for sub in $SUBS; do
  out=$ROOT/derivatives/$sub; mkdir -p $out
  T1=$ROOT/data/$sub/t1/highres001_full_normfilter.nii.gz
  if [ ! -f $out/t1_brain.nii.gz ]; then
    bet $T1 $out/t1_brain -f 0.4 -R
    flirt -in $out/t1_brain -ref $MNI -omat $out/t12mni.mat -dof 12 -out $out/t1_mni
  fi
  for run in task001_run001 task001_run002; do
    o=$out/$run; mkdir -p $o
    [ -f $o/bold_mni.nii.gz ] && continue
    B=$ROOT/data/$sub/bold/$run/bold.nii.gz
    mcflirt -in $B -out $o/bold_mc -plots -refvol 0 -rmsrel -rmsabs
    fslroi $o/bold_mc $o/ref 0 1
    bet $o/ref $o/ref_brain -f 0.3 -F
    fslmaths $o/bold_mc -mas $o/ref_brain $o/bold_mc_brain
    flirt -in $o/ref_brain -ref $out/t1_brain -omat $o/epi2t1.mat -dof 6
    convert_xfm -omat $o/epi2mni.mat -concat $out/t12mni.mat $o/epi2t1.mat
    flirt -in $o/bold_mc_brain -ref $MNI -applyxfm -init $o/epi2mni.mat -out $o/bold_mni -interp trilinear
  done
  echo "done $sub"
done
