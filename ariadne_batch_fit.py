#!/usr/bin/env python
"""
ARIADNE Batch Fitting Script

Run SED fitting on multiple targets with configurable parameters and optional plotting.

Usage:
    python ariadne_batch_fit.py --csv-path <path> [options]

Example:
    python ariadne_batch_fit.py \\
        --csv-path photometry.csv \\
        --nlive 50 \\
        --n-samples 1000 \\
        --dlogz 0.999 \\
        --threads 48 \\
        --plot-sed \\
        --plot-corner \\
        --num-targets 5
"""

import sys
import os
import argparse
import pandas as pd
from pathlib import Path

# Add the local astroARIADNE path to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from astroARIADNE.star import Star
from astroARIADNE.fitter import Fitter
from astroARIADNE.plotter import SEDPlotter


# Photometric bands configuration
COL_TO_ARIADNE_FIT = {
    # GALEX
    "FUV": "GALEX_FUV",
    "NUV": "Galex_NUV",
    # Tycho
    "BT": "TYCHO_B_MvB",
    "VT": "TYCHO_V_MvB",
    # Johnson ground-based
    "B": "GROUND_JOHNSON_B",
    "V": "GROUND_JOHNSON_V",
    # Gaia
    "G": "GaiaDR2v2_G",
    "BP": "GaiaDR2v2_BP",
    "RP": "GaiaDR2v2_RP",
    # 2MASS
    "J": "2MASS_J",
    "H": "2MASS_H",
    "K": "2MASS_Ks",
    # WISE
    "W1": "WISE_RSR_W1",
    "W2": "WISE_RSR_W2",
}

COL_TO_ARIADNE_IR = {
    "W3": "WISE_RSR_W3",
    "W4": "WISE_RSR_W4",
}


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ARIADNE SED fitting on multiple targets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with defaults
  python ariadne_batch_fit.py --csv-path photometry.csv

  # With custom fitting parameters
  python ariadne_batch_fit.py --csv-path photometry.csv \\
    --nlive 100 --n-samples 2000 --dlogz 0.995 --threads=1

  # With all plotting enabled
  python ariadne_batch_fit.py --csv-path photometry.csv \\
    --plot-sed --plot-sed-no-model --plot-bma-hist --plot-corner

  # Process 3 targets with minimal plotting
  python ariadne_batch_fit.py --csv-path photometry.csv \\
    --num-targets 3 --plot-sed
        """,
    )

    # Required arguments
    parser.add_argument(
        "--csv-path", required=True, help="Path to CSV file with photometry data"
    )

    # Fitting parameters
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads for dynesty (default: 1). Increase for better efficiency.",
    )
    parser.add_argument(
        "--nlive",
        type=int,
        default=50,
        help="Number of live points for dynesty (default: 50). Increase for better accuracy.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples for posterior estimation (default: 1000)",
    )
    parser.add_argument(
        "--dlogz",
        type=float,
        default=0.999,
        help="Log evidence tolerance for dynesty (default: 0.999). Smaller = more accurate.",
    )

    # Data selection
    parser.add_argument(
        "--target-col",
        default="RKS_name",
        help="Name of target identifier column (default: RKS_name)",
    )
    parser.add_argument(
        "--num-targets",
        type=int,
        default=10,
        help="Number of targets to process (default: 10)",
    )
    parser.add_argument(
        "--target-list",
        nargs="+",
        help="Specific target names to process (overrides --num-targets)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default="ARIADNE.outputs",
        help="Output directory for results (default: ARIADNE.outputs)",
    )

    # Plotting options (all optional)
    parser.add_argument("--plot-sed", action="store_true", help="Plot SED with model")
    parser.add_argument(
        "--plot-sed-no-model", action="store_true", help="Plot SED without model"
    )
    parser.add_argument(
        "--plot-bma-hist", action="store_true", help="Plot BMA histogram"
    )
    parser.add_argument(
        "--plot-corner", action="store_true", help="Plot corner diagram"
    )
    parser.add_argument(
        "--plot-all", action="store_true", help="Enable all plotting options"
    )

    # Model selection
    parser.add_argument(
        "--models",
        nargs="+",
        default=["phoenix", "btsettl", "btnextgen", "btcond", "kurucz", "ck04"],
        help="Model grids to use in fitting (default: phoenix btsettl btnextgen btcond kurucz ck04)",
    )

    # Other options
    parser.add_argument(
        "--av-fixed",
        type=float,
        default=0.0,
        help="Fixed extinction Av value (default: 0.0)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def build_mag_dict(row, col_mapping):
    """Build magnitude dictionary from a DataFrame row."""
    mag_dict = {}
    for csv_mag_col, ar_filter in col_mapping.items():
        err_col = f"e_{csv_mag_col}"
        if csv_mag_col not in row.index or err_col not in row.index:
            continue
        mag = row[csv_mag_col]
        err = row[err_col]
        if pd.isna(mag) or pd.isna(err):
            continue
        err = float(err)
        if err <= 0:
            continue
        mag_dict[ar_filter] = (float(mag), err)
    return mag_dict


def process_target(target, row, args):
    """Process a single target."""
    print(f"\n{'='*60}")
    print(f"Processing target: {target}")
    print(f"{'='*60}\n")

    # Extract coordinates and parallax
    ra = float(row["ra_deg"])
    dec = float(row["dec_deg"])
    plx = float(row["pi"])  # mas
    plx_e = float(row["pi_e"])  # mas

    # Build magnitude dictionary
    mag_dict = build_mag_dict(row, COL_TO_ARIADNE_FIT)

    if len(mag_dict) == 0:
        print(f"[WARNING] No valid photometry for {target}. Skipping.")
        return False

    print(f"Using {len(mag_dict)} bands for fit:")
    for f, (m, e) in mag_dict.items():
        print(f"  {f}: {m:.3f} ± {e:.3f}")

    # Create Star object (offline mode)
    try:
        s = Star(
            str(target),
            ra,
            dec,
            plx=plx,
            plx_e=plx_e,
            Av=args.av_fixed,
            Av_e=0.0,
            offline=True,
            mag_dict=mag_dict,
            verbose=args.verbose,
        )
        print(f"[OK] Star created: {s.starname if hasattr(s, 'starname') else s.name}")
        print(f"[OK] Using parallax (mas): {plx} ± {plx_e}")
    except Exception as e:
        print(f"[ERROR] Failed to create Star object: {e}")
        return False

    # Setup output folder
    out_folder = os.path.join(args.output_dir, str(target))
    os.makedirs(out_folder, exist_ok=True)

    # Configure fitter
    try:
        f = Fitter()
        f.star = s
        f.setup = ["dynesty", args.nlive, args.dlogz, "multi", "rwalk", args.threads, False]
        f.av_law = "fitzpatrick"
        f.out_folder = out_folder
        f.bma = True
        f.models = args.models
        f.n_samples = args.n_samples

        f.prior_setup = {
            "teff": ("default"),
            "logg": ("default"),
            "z": ("default"),
            "dist": ("default"),
            "rad": ("default"),
            "Av": ("fixed", args.av_fixed),
        }

        print("[OK] Fitter configured.")
        print(f"  out_folder: {f.out_folder}")
        print(f"  nlive: {args.nlive}")
        print(f"  n_samples: {args.n_samples}")
        print(f"  dlogz: {args.dlogz}")
        print(f"  models: {', '.join(args.models)}")
        print(f"  threads: {args.threads}")

        # Run fitting
        print("\n[INFO] Initializing fitter...")
        f.initialize()
        print("[INFO] Running BMA fit (this may take a while)...")
        f.fit_bma()
        print("[OK] Fitting completed successfully.")

    except Exception as e:
        print(f"[ERROR] Fitting failed: {e}")
        return False

    # Generate plots if requested
    if any(
        [
            args.plot_sed,
            args.plot_sed_no_model,
            args.plot_bma_hist,
            args.plot_corner,
            args.plot_all,
        ]
    ):
        try:
            in_file = os.path.join(out_folder, "BMA.pkl")
            plots_out_folder = os.path.join(out_folder, "plots")
            os.makedirs(plots_out_folder, exist_ok=True)

            # Set ARIADNE_MODELS environment variable
            model_dir = str(SCRIPT_DIR)
            if not os.path.isdir(model_dir):
                print(f"[WARNING] ARIADNE models directory not found: {model_dir}")
            os.environ["ARIADNE_MODELS"] = model_dir

            artist = SEDPlotter(in_file, plots_out_folder, model="phoenix")

            if args.plot_sed_no_model or args.plot_all:
                print("  Plotting SED (no model)...")
                artist.plot_SED_no_model()

            if args.plot_sed or args.plot_all:
                print("  Plotting SED...")
                artist.plot_SED()

            if args.plot_bma_hist or args.plot_all:
                print("  Plotting BMA histogram...")
                artist.plot_bma_hist()

            if args.plot_corner or args.plot_all:
                print("  Plotting corner diagram...")
                artist.plot_corner()

            print(f"[OK] Plots saved to: {plots_out_folder}")

        except Exception as e:
            print(f"[WARNING] Plotting failed: {e}")

    return True


def main():
    """Main function."""
    args = parse_arguments()

    # Validate CSV file
    if not os.path.isfile(args.csv_path):
        print(f"[ERROR] CSV file not found: {args.csv_path}")
        sys.exit(1)

    # Load photometry data
    try:
        df = pd.read_csv(args.csv_path)
        print(f"[OK] Loaded {len(df)} targets from {args.csv_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load CSV: {e}")
        sys.exit(1)

    # Get target list
    if args.target_list:
        target_list = args.target_list
        print(f"[INFO] Processing {len(target_list)} specified targets")
    else:
        target_list = df[args.target_col].head(args.num_targets).astype(str).tolist()
        print(f"[INFO] Processing first {len(target_list)} targets")

    # Process targets
    processed = 0
    failed = 0

    for i, target in enumerate(target_list, 1):
        print(f"\n[{i}/{len(target_list)}] ", end="")

        try:
            row = df.loc[df[args.target_col].astype(str) == str(target)].iloc[0]
        except IndexError:
            print(f"[ERROR] Target {target} not found in CSV. Skipping.")
            failed += 1
            continue

        success = process_target(target, row, args)
        if success:
            processed += 1
        else:
            failed += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"  Successful: {processed}/{len(target_list)}")
    print(f"  Failed: {failed}/{len(target_list)}")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'='*60}\n")

    if processed == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
