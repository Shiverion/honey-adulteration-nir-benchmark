"""Exercise the blinded workflow on archived Auckland spectra and surrogate lots."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from followup import LABEL_FIELDS, SCAN_FIELDS, json_bytes, sha256, write_new

import numpy as np
import pandas as pd

from datasets import GRID_NM, get_file, load_auckland


INSTRUMENT = "SOC-710"
DEVELOPMENT_BRANDS = ["C1", "C10", "C4", "C5"]
CALIBRATION_BRAND = "C8"
TEST_BRANDS = ["C7", "C9"]


def run_stage(command, *args):
    cli = Path(__file__).with_name("followup.py")
    subprocess.run([sys.executable, str(cli), command, *map(str, args)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a new output path for the dry run.")
    data = load_auckland(get_file(args.data, "auckland", args.download))
    rows = data.meta.assign(row=np.arange(len(data.x)), instrument=INSTRUMENT, dose_pct=data.meta["nominal_pct"])
    rows["surrogate"] = rows[["group", "product", "acquisition"]].astype(str).agg("|".join, axis=1)
    rows["preparation_id"] = [f"P{i:04d}" for i in rows["row"]]
    rows["scan_id"] = rows["preparation_id"]
    dev = rows.loc[rows["group"].isin(DEVELOPMENT_BRANDS)].assign(partition="development", lot_id=lambda r: r["group"])
    cal = rows.loc[rows["group"].eq(CALIBRATION_BRAND) & rows["dose_pct"].eq(0)].assign(
        partition="calibration", lot_id=lambda r: r["surrogate"],
    )
    test = rows.loc[rows["group"].isin(TEST_BRANDS) & rows["dose_pct"].isin([0, 5, 10])].drop_duplicates(["surrogate", "dose_pct"])
    test = test.loc[test.groupby("surrogate")["dose_pct"].transform("size").eq(3)].assign(
        partition="test", lot_id=lambda r: r["surrogate"],
    )
    fit = pd.concat([dev, cal], ignore_index=True)
    manifest = pd.concat([fit, test])[["lot_id", "partition"]].drop_duplicates()
    band_names = [f"{w:.10f}nm" for w in GRID_NM]
    with TemporaryDirectory() as directory:
        tmp = Path(directory)
        manifest.to_csv(tmp / "partitions.csv", index=False)
        for phase, table in (("fit", fit), ("test", test)):
            pd.concat([
                table[SCAN_FIELDS].reset_index(drop=True),
                pd.DataFrame(data.x[table["row"]], columns=band_names),
            ], axis=1).to_csv(tmp / f"{phase}-spectra.csv", index=False)
            table[LABEL_FIELDS].to_csv(tmp / f"{phase}-labels.csv", index=False)
        run_stage(
            "freeze", "--partitions", tmp / "partitions.csv", "--labels", tmp / "fit-labels.csv",
            "--spectra", tmp / "fit-spectra.csv", "--source-instrument", INSTRUMENT,
            "--adulterants", "sugar syrup", "--out", tmp / "frozen.pkl",
        )
        run_stage("score", "--frozen", tmp / "frozen.pkl", "--spectra", tmp / "test-spectra.csv", "--out", tmp / "blind.json")
        digest = sha256((tmp / "blind.json").read_bytes())
        run_stage(
            "evaluate", "--receipt", tmp / "frozen.json", "--scores", tmp / "blind.json",
            "--expected-scores-sha256", digest, "--labels", tmp / "test-labels.csv", "--out", tmp / "decision.json",
        )
        report = json.loads((tmp / "decision.json").read_bytes())
    report["retrospective_dry_run"] = True
    report["dry_run_design"] = {
        "source": "Auckland public dataset", "development_brands": DEVELOPMENT_BRANDS,
        "calibration_brand": CALIBRATION_BRAND, "test_brands": TEST_BRANDS,
        "test_unit": "Brand, product and acquisition surrogate with unverified parent lots.",
        "selection": "Keep complete nominal 0/5/10 dose panels, with the first sorted preparation key per dose.",
    }
    write_new(args.out, json_bytes(report))


if __name__ == "__main__":
    main()
