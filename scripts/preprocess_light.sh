#!/bin/bash
# Lighter FSL pipeline for screening many subjects (~150 MB and ~5 min per subject).
# Reads data/OneDrive_1_9-23-2026/<sub>/, writes derivatives/<sub>/ in the same layout as preprocess_fsl.sh
# but keeps only: t1_brain, t1_mni, and per run bold_mc (int16), bold_mc.par, bold_mni (3 mm, int16), epi2mni.mat.
# usage: scripts/preprocess_light.sh [sub ...]   (default: every subject in the download that has a T1 and no derivatives yet)
export FSLOUTPUTTYPE=NIFTI_GZ
ROOT=/Users/cmahlen/random_code_projects/paul_mri_class
SRC=$ROOT/data/OneDrive_1_9-23-2026
MNI=$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz
SUBS=${@:-$(ls $SRC)}
for sub in $SUBS; do
  T1=$SRC/$sub/t1/highres001_full_normfilter.nii.gz
  [ -f $T1 ] || { echo "skip $sub (no T1)"; continue; }
  out=$ROOT/derivatives/$sub
  [ -f $out/task001_run002/bold_mni.nii.gz ] && { echo "skip $sub (done)"; continue; }
  mkdir -p $out
  bet $T1 $out/t1_brain -f 0.4 -R
  flirt -in $out/t1_brain -ref $MNI -omat $out/t12mni.mat -dof 12 -out $out/t1_mni
  for run in task001_run001 task001_run002; do
    o=$out/$run; mkdir -p $o
    B=$SRC/$sub/bold/$run/bold.nii.gz
    [ -f $B ] || { echo "skip $sub $run (no bold)"; continue; }
    mcflirt -in $B -out $o/mc_tmp -plots -refvol 0
    mv $o/mc_tmp.par $o/bold_mc.par
    fslroi $o/mc_tmp $o/ref 0 1
    bet $o/ref $o/ref_brain -f 0.3 -F
    flirt -in $o/ref_brain -ref $out/t1_brain -omat $o/epi2t1.mat -dof 6
    convert_xfm -omat $o/epi2mni.mat -concat $out/t12mni.mat $o/epi2t1.mat
    fslmaths $o/mc_tmp -mas $o/ref_brain_mask $o/mc_brain_tmp
    flirt -in $o/mc_brain_tmp -ref $MNI -applyisoxfm 3 -init $o/epi2mni.mat -out $o/bold_mni -interp trilinear -datatype short
    fslmaths $o/mc_tmp $o/bold_mc -odt short
    rm -f $o/mc_tmp.nii.gz $o/mc_brain_tmp.nii.gz $o/ref.nii.gz $o/ref_brain.nii.gz $o/ref_brain_mask.nii.gz $o/epi2t1.mat
  done
  echo "done $sub $(date +%H:%M)"
done
