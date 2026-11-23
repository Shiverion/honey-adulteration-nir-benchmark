# 24-lot honey screening feasibility pilot

The pilot checks whether the intended instrument can support the proposed screening targets before the larger blinded study. The working plan uses **24 independent honey lots, 120 preparations and 30 second-day rescans**. It compares SVM, XGBoost and MLP at a threshold calibrated toward 90% specificity, with sensitivity targets of 80% at each dose. The pilot’s estimates guide measurement and model development. The [final validation protocol](FOLLOWUP.md) applies a separate statistical decision rule to new lots.

## Check an export before preparing samples

Use the instrument’s native wavelength columns, named in nanometers like `980.0nm`. Arrange the export as one row per scan, with `lot_id,preparation_id,instrument,scan_id` followed by the measured wavelength columns. The four IDs can be added from the scan schedule when the instrument does not export them. Use the same signal representation and instrument ID across days.

```powershell
python pilot.py check-export --spectra pilot_data/first_export.csv
```

The check accepts a single scan and reports the native span, number of channels, and whether the device covers 780–1000 nm. It rejects duplicate scan IDs, non-finite signals and constant spectra. Save the original export and carry the native columns into the analysis CSV. A paired honey spectroscopy study found separation over wavelengths outside the benchmark window. This pilot can test whether the instrument’s wider coverage helps on the same preparations. [Bilgin et al.](https://doi.org/10.3390/s26092750)

## Prepare and scan the pilot

The [working plan](results/pilot_plan/design.json) uses `screening-unit`, rice and sucrose, and placeholder lot IDs. For collection, create `pilot_data/lot_ids.csv` with one `lot_id` column and 24 stable, traceable parent IDs. Keep the same identifiers in all later study inventories. Choose the syrup names for the intended screening use.

```powershell
python pilot.py prepare --lot-ids pilot_data/lot_ids.csv --instrument screening-unit --adulterants rice sucrose --repeat-lots 6 --out pilot_run/design
```

- Use [`preparation_key.csv`](results/pilot_plan/preparation_key.csv) to prepare one pure control per lot and one sample for each syrup at 5% and 10%. Use a consistent concentration basis, such as mass percent of syrup in the final mixture.
- Give instrument operators the [`scan_schedule.csv`](results/pilot_plan/scan_schedule.csv). It contains opaque preparation IDs and independently randomized run orders for the two days. Store the preparation key separately. Record deviations from the planned order and any changes in acquisition conditions.
- On the second day, rescan the same 30 preparations from the six scheduled lots. Keep their preparation IDs, and use the distinct second-day scan IDs. Save each day’s export as a separate CSV with the four ID columns and the native wavelength columns.

## Analyze the pilot

```powershell
python pilot.py analyze --plan pilot_run/design --baseline-spectra pilot_data/day_1.csv --repeat-spectra pilot_data/day_2.csv --out pilot_run/analysis
```

The inputs are matched to the scheduled scan IDs and the planned dose panel. Omit `--repeat-spectra` for an initial analysis while second-day scans are pending. Use a new output directory when those scans arrive.

With 24 independent lots, each of six folds holds out four lots for testing, uses 10 separate lots to calibrate each decision threshold, and uses 10 for development and three-fold grouped model tuning. Standardization is fitted on development data. Repeat-day comparisons use the same frozen fit and threshold as the first-day score for that preparation. If several lots share a higher-level sampling source that needs to stay together, supply a `lot_id,validation_group` CSV with `--validation-groups` and configure `--outer-folds` and `--calibration-groups` to leave at least three development groups per fold.

Both spectral windows use 48 evenly spaced features and identical splits. One window spans the device’s native wavelength range. The 780–1000 nm window is evaluated when fully covered and distinct from the native span. The tool reports held-out specificity, sensitivity at each dose, and per-syrup detection. For the second day it reports decision flips and the root-mean-square difference between paired spectra after standard normal variate normalization. A lot counts as a detection at a dose only if every planned syrup at that dose is detected. Read 5% sensitivity alongside realized held-out specificity. The point-target flag summarizes pilot performance for planning the separate blinded test.

If the pilot supports both low-dose sensitivity and specificity, update the assumed rates in `followup.py plan` before sizing the definitive test. If 5% sensitivity stays low, review repeatability and per-syrup results before changing acquisition or calibration. Use `--pilot-lots pilot_run/design/lot_inventory.csv` when freezing the blinded study to exclude those IDs from its test partition.

## Verification with archived data

Run `python -m unittest discover -s tests` to check the schedule, disjoint folds, export checks, paired scans, and pilot-lot exclusion. The [Auckland demonstration](results/pilot_archive_demo/summary.json) checks both windows using a single syrup and 36 product–acquisition surrogates. It uses two brands for development tuning, one for calibration and one for testing in each fold. The archive has no verified repeat-day records or independent parent-lot IDs. Reproduce it with `python pilot_demo.py --out reproduced/pilot_archive_demo`.
