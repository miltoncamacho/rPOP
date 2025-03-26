#!/usr/bin/env python3
"""
rPOP (Python Version) - Emulates the original MATLAB rPOP code that uses:
  1) SPM Old Normalization
  2) AFNI 3dFWHMx
  3) Differential smoothing to 10mm FWHM
  4) Logging to CSV
Arguments are passed via argparse instead of spm_select/input.
Logging is used instead of print statements.
"""

import os
import sys
import csv
import math
import argparse
import logging
import datetime
import subprocess

import nibabel as nib
import numpy as np

from nipype.interfaces.spm import Normalize, Smooth
from nipype.interfaces.matlab import MatlabCommand

###############################################################################
# Logging Configuration
###############################################################################
# Configure logging to output messages to the console with timestamps
logger = logging.getLogger("rpop")
logger.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
ch.setFormatter(formatter)
logger.addHandler(ch)

###############################################################################
# Function to reset NIfTI origin to center
###############################################################################
def reset_origin_to_center(nifti_file):
    """
    Reset image origin to the center of the volume.
    Equivalent to MATLAB's spm_get_space with inverse voxel shift.
    """
    logger.info(f"Resetting origin to center: {nifti_file}")
    img = nib.load(nifti_file)
    hdr = img.header.copy()
    aff = img.affine.copy()
    dims = hdr.get_data_shape()
    center_vox = [(d + 1) / 2.0 for d in dims[:3]]
    voxel_sizes = hdr.get_zooms()[:3]
    aff[:3, 3] = -np.multiply(center_vox, voxel_sizes)
    new_img = nib.Nifti1Image(img.get_fdata(), aff, hdr)
    nib.save(new_img, nifti_file)

###############################################################################
# Main rPOP Logic
###############################################################################
def main():
    # Argument parser to replace user input and spm_select from MATLAB
    parser = argparse.ArgumentParser(
        description="Python rPOP using Nipype (SPM old normalization + AFNI FWHMx) and logging."
    )
    parser.add_argument("-v", "--volumes", nargs="+", required=True,
                        help="Paths to 3D NIfTI volumes to process (mimics spm_select(Inf,'image')).")
    parser.add_argument("-o", "--outdir", default=".",
                        help="Output directory (tables/logs). Default: current directory")
    parser.add_argument("-a", "--afni-fwhmx", default="3dFWHMx",
                        help="Path or command name for AFNI's 3dFWHMx. Default='3dFWHMx'")
    parser.add_argument("--origin-reset", action="store_true",
                        help="If set, resets the origin of each volume to its center before normalization.")
    parser.add_argument("--vox-size", default="2,2,2",
                        help="Comma-separated voxel size for writing normalized images (e.g., '2,2,2'). Default=2,2,2")
    parser.add_argument("--bbox", default="-100,-130,-80,100,100,110",
                        help="Comma-separated bounding box for writing normalized images. Default='-100,-130,-80,100,100,110'")
    parser.add_argument("--target-fwhm", type=float, default=10.0,
                        help="Target FWHM after differential smoothing. Default=10.0 mm")
    parser.add_argument("--spm-cmd", default=None,
                        help="Path to matlab or MCR-based SPM12 if needed. If not set, uses environment defaults.")
    parser.add_argument("--template-choice", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Which warping template set to use:\n 1 = Tracer-independent (ALL TEMPLATES)\n 2 = 18F-florbetapir\n 3 = 18F-florbetaben\n 4 = 18F-flutemetamol\n Default=1")

    args = parser.parse_args()

    # Configure SPM command if provided
    if args.spm_cmd:
        MatlabCommand.set_default_matlab_cmd(args.spm_cmd)

    # Parse and validate voxel size
    try:
        vox_size = list(map(float, args.vox_size.split(",")))
        if len(vox_size) != 3:
            raise ValueError
    except ValueError:
        logger.error("--vox-size must be three comma-separated numbers, e.g. '2,2,2'")
        sys.exit(1)

    # Parse and validate bounding box
    try:
        bbox_vals = list(map(float, args.bbox.split(",")))
        if len(bbox_vals) != 6:
            raise ValueError
        bounding_box = [bbox_vals[:3], bbox_vals[3:6]]
    except ValueError:
        logger.error("--bbox must be six comma-separated numbers, e.g. '-100,-130,-80,100,100,110'")
        sys.exit(1)

    logger.info(f"Using voxel size: {vox_size}")
    logger.info(f"Using bounding box: {bounding_box}")
    logger.info(f"Target FWHM: {args.target_fwhm}")

    # Set up template paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tdir = os.path.join(script_dir, "templates")

    # List of all templates grouped by tracer
    tfbpall  = [os.path.join(tdir, "Template_FBP_all.nii")]
    tfbppos  = [os.path.join(tdir, "Template_FBP_pos.nii")]
    tfbpneg  = [os.path.join(tdir, "Template_FBP_neg.nii")]
    tfbball  = [os.path.join(tdir, "Template_FBB_all.nii")]
    tfbbpos  = [os.path.join(tdir, "Template_FBB_pos.nii")]
    tfbbneg  = [os.path.join(tdir, "Template_FBB_neg.nii")]
    tfluteall= [os.path.join(tdir, "Template_FLUTE_all.nii")]
    tflutepos= [os.path.join(tdir, "Template_FLUTE_pos.nii")]
    tfluteneg= [os.path.join(tdir, "Template_FLUTE_neg.nii")]

    # Combine template sets by tracer
    warptempl_fbp = tfbpall + tfbppos + tfbpneg
    warptempl_fbb = tfbball + tfbbpos + tfbbneg
    warptempl_flute = tfluteall + tflutepos + tfluteneg
    warptempl_all = warptempl_fbp + warptempl_fbb + warptempl_flute

    # Select templates based on user input
    if args.template_choice == 1:
        warptempl = warptempl_all
    elif args.template_choice == 2:
        warptempl = warptempl_fbp
    elif args.template_choice == 3:
        warptempl = warptempl_fbb
    else:
        warptempl = warptempl_flute

    logger.info(f"Selected template set: {warptempl}")

    # Set up SPM Normalize interface
    norm = Normalize()
    norm.inputs.jobtype = "estwrite"
    norm.inputs.write_bounding_box = bounding_box
    norm.inputs.write_voxel_sizes = vox_size
    norm.inputs.write_wrap = [0, 0, 0]
    norm.inputs.write_interp = 1
    norm.inputs.out_prefix = "w"
    norm.inputs.template = warptempl

    # Set up SPM Smooth interface
    smoother = Smooth()
    smoother.inputs.out_prefix = "s"

    # Prepare CSV log tables
    dbests = []
    dbwarn = []

    # Loop through input NIfTI volumes
    for vol in args.volumes:
        vol = os.path.abspath(vol)
        if not os.path.isfile(vol):
            logger.warning(f"Skipping non-existent file: {vol}")
            continue

        if args.origin_reset:
            reset_origin_to_center(vol)

        norm.inputs.source = vol
        norm.inputs.apply_to_files = [vol]

        logger.info(f"Normalizing: {vol}")
        try:
            res = norm.run()
        except Exception as e:
            logger.error(f"Normalization failed: {e}")
            continue

        # Get warped image from SPM output
        warped_file = res.outputs.normalized_files[0]
        base = os.path.splitext(warped_file)[0]
        txtfwhm = base + "_automask.txt"

        # Estimate FWHM with 3dFWHMx and -2difMAD
        cmd = [args.afni_fwhmx, "-automask", "-2difMAD", "-input", warped_file, "-out", txtfwhm]
        subprocess.run(cmd)

        if not os.path.exists(txtfwhm):
            logger.error(f"FWHM output missing: {txtfwhm}")
            continue

        # Read estimated FWHM values
        with open(txtfwhm, "r") as f:
            lines = f.read().strip().split()

        try:
            fx, fy, fz = map(float, lines[:3])
        except Exception as e:
            logger.error(f"Could not parse FWHM: {e}")
            continue

        # Check for abnormal FWHM and re-run without -2difMAD if needed
        rerun_flag = "0"
        if max(fx, fy, fz) > 25:
            logger.warning(f"High FWHM detected. Re-running without -2difMAD.")
            rerun_flag = "1"
            txtfwhm_mod = base + "_automask_mod.txt"
            cmd_mod = [args.afni_fwhmx, "-automask", "-input", warped_file, "-out", txtfwhm_mod]
            subprocess.run(cmd_mod)
            with open(txtfwhm_mod, "r") as f:
                lines = f.read().strip().split()
                fx, fy, fz = map(float, lines[:3])
                dbwarn.append(f"High FWHM rerun: {warped_file}")

        # Compute differential smoothing to target FWHM
        def fwhm_filter(est, target):
            return 0 if est > target else math.sqrt(target**2 - est**2)

        filtx = fwhm_filter(fx, args.target_fwhm)
        filty = fwhm_filter(fy, args.target_fwhm)
        filtz = fwhm_filter(fz, args.target_fwhm)

        # Smooth the image using calculated filters
        smoother.inputs.in_files = warped_file
        smoother.inputs.fwhm = [filtx, filty, filtz]
        smoother.run()

        # Log output data for CSV
        dbests.append([
            warped_file,
            f"{fx:.4f}", f"{fy:.4f}", f"{fz:.4f}",
            f"{filtx:.4f}", f"{filty:.4f}", f"{filtz:.4f}", rerun_flag
        ])

    # Write main CSV
    timestamp = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    main_csv = os.path.join(args.outdir, f"rPOP_{timestamp}.csv")
    with open(main_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Filename", "EstimatedFWHMx", "EstimatedFWHMy", "EstimatedFWHMz",
            "FWHMfilterappliedx", "FWHMfilterappliedy", "FWHMfilterappliedz",
            "AFNIEstimationRerunMod"
        ])
        writer.writerows(dbests)

    # Write warnings if needed
    if dbwarn:
        warn_csv = os.path.join(args.outdir, f"rPOPWarnings_{timestamp}.csv")
        with open(warn_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Warning"])
            for w in dbwarn:
                writer.writerow([w])

    logger.info("rPOP complete. Check CSV logs and smoothed images.")

if __name__ == "__main__":
    main()
