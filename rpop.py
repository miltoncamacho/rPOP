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
from nipype.interfaces.afni import FWHMx
from nipype.interfaces.matlab import MatlabCommand


###############################################################################
# Logging Configuration
###############################################################################
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
# (Equivalent to F. Yamashita code from the old script)
###############################################################################
def reset_origin_to_center(nifti_file):
    """
    Approximate 'spm_get_space(..., inv(vs))' by shifting the affine
    so that the volume center is at the origin.
    Overwrites the original file for simplicity.
    """
    logger.info(f"Resetting origin to center: {nifti_file}")
    img = nib.load(nifti_file)
    hdr = img.header.copy()
    aff = img.affine.copy()
    dims = hdr.get_data_shape()

    # Center in voxel space
    center_vox = [(d + 1) / 2.0 for d in dims[:3]]
    # Adjust the affine so 0,0,0 in world space is the center of the volume
    voxel_sizes = hdr.get_zooms()[:3]
    # old SPM code does: vs(1:3,4) = (st.vol.dim+1)/2
    # then spm_get_space(..., inv(vs))
    # We'll approximate that:
    aff[:3, 3] = -np.multiply(center_vox, voxel_sizes)

    new_img = nib.Nifti1Image(img.get_fdata(), aff, hdr)
    nib.save(new_img, nifti_file)


###############################################################################
# Main rPOP Logic
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Python rPOP using Nipype (SPM old normalization + AFNI FWHMx) and logging."
    )
    parser.add_argument(
        "-v", "--volumes", nargs="+", required=True,
        help="Paths to 3D NIfTI volumes to process (mimics spm_select(Inf,'image'))."
    )
    parser.add_argument(
        "-o", "--outdir", default=".",
        help="Output directory (tables/logs). Default: current directory"
    )
    parser.add_argument(
        "-a", "--afni-fwhmx", default="3dFWHMx",
        help="Path or command name for AFNI's 3dFWHMx. Default='3dFWHMx'"
    )
    parser.add_argument(
        "--origin-reset", action="store_true",
        help="If set, reset the origin to center for all volumes."
    )
    parser.add_argument(
        "--template-choice", type=int, default=1, choices=[1,2,3,4],
        help=(
            "Which warping template set to use:\n"
            " 1 = Tracer-independent (ALL TEMPLATES)\n"
            " 2 = 18F-florbetapir\n"
            " 3 = 18F-florbetaben\n"
            " 4 = 18F-flutemetamol\n"
            "Default=1"
        )
    )
    args = parser.parse_args()

    # Welcome messages
    logger.info("********** Welcome to rPOP (Python) **********")
    logger.info("Depends on SPM12, AFNI, Nipype, and Python. "
                "Academic/research only; no warranty. Not for clinical use.")

    # Prepare output folder
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    # Volumes
    volumes = [os.path.abspath(v) for v in args.volumes]
    logger.info(f"Input volumes: {volumes}")

    # AFNI 3dFWHMx path
    afnifx = args.afni_fwhmx
    logger.info(f"Using AFNI 3dFWHMx at: {afnifx}")

    # Prepare the sets of templates from the original rPOP logic
    # We replicate the 'templates/' folder structure and naming
    # Typically, you'd find these next to 'rPOP' in a 'templates' folder
    # For demonstration, let's assume they're in ./templates relative to script
    # Adjust as needed or add an argument to specify the template directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tdir = os.path.join(script_dir, "templates")

    tfbpall  = [os.path.join(tdir, "Template_FBP_all.nii")]
    tfbppos  = [os.path.join(tdir, "Template_FBP_pos.nii")]
    tfbpneg  = [os.path.join(tdir, "Template_FBP_neg.nii")]

    tfbball  = [os.path.join(tdir, "Template_FBB_all.nii")]
    tfbbpos  = [os.path.join(tdir, "Template_FBB_pos.nii")]
    tfbbneg  = [os.path.join(tdir, "Template_FBB_neg.nii")]

    tfluteall= [os.path.join(tdir, "Template_FLUTE_all.nii")]
    tflutepos= [os.path.join(tdir, "Template_FLUTE_pos.nii")]
    tfluteneg= [os.path.join(tdir, "Template_FLUTE_neg.nii")]

    warptempl_fbp = tfbpall + tfbppos + tfbpneg
    warptempl_fbb = tfbball + tfbbpos + tfbbneg
    warptempl_flute = tfluteall + tflutepos + tfluteneg
    warptempl_all = warptempl_fbp + warptempl_fbb + warptempl_flute

    # Based on user input
    if args.template_choice == 1:
        warptempl = warptempl_all
    elif args.template_choice == 2:
        warptempl = warptempl_fbp
    elif args.template_choice == 3:
        warptempl = warptempl_fbb
    else:
        warptempl = warptempl_flute

    logger.info(f"Selected template set (# {args.template_choice}): {warptempl}")

    # We'll store data for final CSV
    dbests = []
    dbwarn = []

    ###########################################################################
    # Nipype config for old normalization (spm.Normalize)
    ###########################################################################
    # This replicates "spm.tools.oldnorm.estwrite" with a bounding box and voxel size
    norm = Normalize()
    norm.inputs.jobtype = "estwrite"
    # bounding box from code: [-100 -130 -80; 100 100 110]
    norm.inputs.write_bounding_box = [[-100, -130, -80], [100, 100, 110]]
    # voxel size from code: [2 2 2]
    norm.inputs.write_voxel_sizes = [2, 2, 2]
    norm.inputs.write_wrap = [0, 0, 0]
    # interp=1 => trilinear in SPM
    norm.inputs.write_interp = 1
    # prefix = 'w'
    norm.inputs.out_prefix = "w"
    # pass the selected templates
    norm.inputs.template = warptempl

    # We'll also use SPM's Smooth for the final smoothing step
    smoother = Smooth()
    smoother.inputs.out_prefix = "s"

    # Now loop over volumes
    for vol in volumes:
        if not os.path.isfile(vol):
            logger.warning(f"Skipping non-existent file: {vol}")
            continue

        # If origin reset is requested, shift it
        if args.origin_reset:
            reset_origin_to_center(vol)

        # Now do the old normalization step on this volume
        # We'll set source=vol, apply it to 'resample'=vol, matching the MATLAB code
        norm.inputs.source = vol
        norm.inputs.apply_to_files = [vol]

        logger.info(f"[Normalization] Warping {vol} via Old Normalization with SPM.")
        try:
            res = norm.run()
        except Exception as e:
            logger.error(f"Normalization failed for {vol}: {e}")
            continue

        # The warped file is typically 'w<vol>.nii'
        warped_files = res.outputs.normalized_files
        if not warped_files:
            logger.warning(f"No warped output found for {vol}. Skipping.")
            continue

        # In principle, SPM might output multiple if resample had multiple
        warped_file = warped_files[0]
        if not os.path.isfile(warped_file):
            logger.warning(f"Warped file missing: {warped_file}")
            continue

        # Next, run AFNI 3dFWHMx with -automask -2difMAD
        # The original code creates an output text file, e.g. wtempimg_automask.txt
        base_no_ext = os.path.splitext(warped_file)[0]
        txtfwhm = f"{base_no_ext}_automask.txt"

        cmd = [
            afnifx,
            "-automask",
            "-2difMAD",
            "-input", warped_file,
            "-out", txtfwhm
        ]
        logger.info(f"[FWHM Estimation] Running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.error(f"3dFWHMx failed for {warped_file}:\n{proc.stderr}")
            continue

        if not os.path.isfile(txtfwhm):
            logger.error(f"FWHM output file not created: {txtfwhm}")
            continue

        # Parse the FWHM results
        try:
            with open(txtfwhm, "r") as f:
                lines = f.read().strip().split()
            tempfwhmx = float(lines[0])
            tempfwhmy = float(lines[1])
            tempfwhmz = float(lines[2])
        except Exception as e:
            logger.error(f"Could not parse FWHM from {txtfwhm}: {e}")
            continue

        # Possibly re-run if too high
        # Original code checks if any dimension > 25
        rerun_flag = "0"
        if (tempfwhmx > 25) or (tempfwhmy > 25) or (tempfwhmz > 25):
            logger.warning(
                f"High FWHM for {warped_file} (x={tempfwhmx:.2f}, y={tempfwhmy:.2f}, z={tempfwhmz:.2f}). "
                "Re-running 3dFWHMx without -2difMAD."
            )
            dbwarn.append(
                f"High FWHM for {os.path.basename(warped_file)}: x={tempfwhmx:.2f}, y={tempfwhmy:.2f}, z={tempfwhmz:.2f}"
            )
            rerun_flag = "1"

            txtfwhm_mod = f"{base_no_ext}_automask_mod.txt"
            cmd_mod = [
                afnifx,
                "-automask",
                "-input", warped_file,
                "-out", txtfwhm_mod
            ]
            proc_mod = subprocess.run(cmd_mod, capture_output=True, text=True)
            if proc_mod.returncode == 0 and os.path.isfile(txtfwhm_mod):
                try:
                    with open(txtfwhm_mod, "r") as fm:
                        lines_mod = fm.read().strip().split()
                    tempfwhmx = float(lines_mod[0])
                    tempfwhmy = float(lines_mod[1])
                    tempfwhmz = float(lines_mod[2])
                except Exception:
                    logger.error("Failed to parse re-run FWHM results.")
        
        # Now compute the differential smoothing filter to achieve 10 mm if < 10
        # If the FWHM is > 10, filter = 0
        def calc_filter_value(est):
            if est > 10:
                return 0.0
            return math.sqrt(10.0**2 - est**2)

        filtx = calc_filter_value(tempfwhmx)
        filty = calc_filter_value(tempfwhmy)
        filtz = calc_filter_value(tempfwhmz)

        # If we had the re-run logic with new FWHM values, the filters reflect those new values
        logger.info(
            f"[Differential Smoothing] For {warped_file}, FWHM=(x={tempfwhmx:.2f}, y={tempfwhmy:.2f}, z={tempfwhmz:.2f}), "
            f"Filters=(x={filtx:.2f}, y={filty:.2f}, z={filtz:.2f})."
        )

        # Smooth the warped image using SPM's Smooth
        smoother.inputs.in_files = warped_file
        smoother.inputs.fwhm = [filtx, filty, filtz]
        try:
            smoother.run()
        except Exception as e:
            logger.error(f"Smoothing failed for {warped_file}: {e}")
            continue

        # Save a record for the CSV
        dbests.append([
            warped_file,
            f"{tempfwhmx:.4f}",
            f"{tempfwhmy:.4f}",
            f"{tempfwhmz:.4f}",
            f"{filtx:.4f}",
            f"{filty:.4f}",
            f"{filtz:.4f}",
            rerun_flag
        ])

    ###########################################################################
    # Save final CSV logs
    ###########################################################################
    timestamp = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    main_csv_path = os.path.join(outdir, f"rPOP_{timestamp}.csv")
    logger.info(f"Writing final results to {main_csv_path}")
    with open(main_csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Filename",
            "EstimatedFWHMx",
            "EstimatedFWHMy",
            "EstimatedFWHMz",
            "FWHMfilterappliedx",
            "FWHMfilterappliedy",
            "FWHMfilterappliedz",
            "AFNIEstimationRerunMod"
        ])
        for row in dbests:
            writer.writerow(row)

    if len(dbwarn) > 0:
        warn_csv_path = os.path.join(outdir, f"rPOPWarnings_{timestamp}.csv")
        logger.warning(f"{len(dbwarn)} warning(s) encountered. See {warn_csv_path}")
        with open(warn_csv_path, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Warning"])
            for w in dbwarn:
                writer.writerow([w])

    logger.info("rPOP just finished! Check the warped+smoothed images and CSV logs.")


###############################################################################
# Entry Point
###############################################################################
if __name__ == "__main__":
    main()
