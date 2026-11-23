"""Run the pilot analysis on the archived Auckland release, grouped by brand."""

import argparse
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

from followup import SCAN_FIELDS, json_bytes, write_new

import numpy as np
import pandas as pd

from datasets import get_file, load_auckland


def run_pilot(command, *args):
    cli = Path(__file__).with_name("pilot.py")
    subprocess.run([sys.executable, str(cli), command, *map(str, args)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a new output path for the archive demonstration.")
    source = get_file(args.data, "auckland", args.download)
    raw_headers = pd.read_csv(source, nrows=0).columns
    bands = [name for name in raw_headers if re.fullmatch(r"\d+\.\d+nm", name)]
    wavelengths = np.array([float(name[:-2]) for name in bands])
    data = load_auckland(source, wavelengths)
    meta = data.meta.assign(row=np.arange(len(data.x)), dose_pct=data.meta["nominal_pct"])
    meta["lot_id"] = meta[["group", "product", "acquisition"]].astype(str).agg("|".join, axis=1)
    panel = meta.loc[meta["dose_pct"].isin([0, 5, 10])].drop_duplicates(["lot_id", "dose_pct"])
    panel = panel.loc[panel.groupby("lot_id")["dose_pct"].transform("size").eq(3)]
    groups = panel[["lot_id", "group"]].drop_duplicates().rename(columns={"group": "validation_group"})

    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        groups[["lot_id"]].to_csv(tmp / "lots.csv", index=False)
        groups.to_csv(tmp / "groups.csv", index=False)
        run_pilot("prepare", "--instrument", "SOC-710", "--adulterants", "sugar syrup", "--repeat-lots", 0,
                  "--lot-ids", tmp / "lots.csv", "--out", tmp / "plan")
        key = pd.read_csv(tmp / "plan" / "preparation_key.csv")
        schedule = pd.read_csv(tmp / "plan" / "scan_schedule.csv")
        joined = schedule.merge(key, on=["lot_id", "preparation_id"], validate="one_to_one")
        joined = joined.merge(panel[["lot_id", "dose_pct", "row"]], on=["lot_id", "dose_pct"], validate="one_to_one")
        spectra = pd.concat([joined[SCAN_FIELDS], pd.DataFrame(data.x[joined["row"]], columns=bands)], axis=1)
        spectra.iloc[:1].to_csv(tmp / "first_export.csv", index=False)
        run_pilot("check-export", "--spectra", tmp / "first_export.csv")
        spectra.to_csv(tmp / "baseline.csv", index=False)
        run_pilot("analyze", "--plan", tmp / "plan", "--baseline-spectra", tmp / "baseline.csv",
                  "--validation-groups", tmp / "groups.csv", "--outer-folds", groups["validation_group"].nunique(),
                  "--calibration-groups", 1, "--inner-folds", 2, "--out", args.out)
    write_new(args.out / "archive_context.json", json_bytes({
        "dataset": "Auckland public hyperspectral release", "runs_only_as_demonstration": True,
        "lot_surrogate": "Brand, product and acquisition key. Parent lot identity is unverified.",
        "validation_group": "Brand, with two development brands, one calibration brand and one test brand in each fold.",
        "second_day": "The archive has no verified day-two repeat scans.",
        "bands": "The native span uses the wavelength column labels in the released CSV.",
    }))


if __name__ == "__main__":
    main()
