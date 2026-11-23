# Blinded validation on new honey lots

This protocol takes SVM, XGBoost and MLP forward from the [public-data benchmark](README.md). It evaluates low-dose detection on a designated screening instrument. XGBoost is the working primary model because it led the existing external balanced-accuracy comparison. SVM and MLP are exploratory comparators. The working acceptance targets are **90% specificity, 80% sensitivity at 5%, and 80% sensitivity at 10%**. Confirm or change these targets and the primary model before collecting the final test set. They are study design choices.

## 1. Set the targets and plan the test set

```powershell
python followup.py plan --specificity 0.90 --sensitivity-5 0.80 --sensitivity-10 0.80 --assumed-specificity 0.96 --assumed-sensitivity-5 0.90 --assumed-sensitivity-10 0.92 --power 0.80
```

The first three fractions are acceptance targets. The three assumed rates represent the performance you expect for planning purposes. The example yields **142 independent test lots** and a conservative joint-power bound of **83.4%** under the stated assumptions. With two syrups, this gives 710 test preparations. The planner uses exact binomial tests at each endpoint and a union bound on failure across endpoints, allowing the dose outcomes from the same lot to be correlated. Development and calibration lots are additional. Plan enough lots to replace unusable lots while maintaining a complete panel, and predefine any replacement rules before scoring.

Each test lot contributes one pure preparation and one preparation for each chosen syrup at each of the 5% and 10% levels. Use a consistent concentration basis, such as mass percent of syrup in the final mixture. With two syrups, that is five preparations per lot. A positive-dose lot counts as detected only when the model detects every prescribed syrup at that dose. This gives one binary response per independent lot per endpoint. The tool also reports sensitivity for each syrup to show where misses occur.

## 2. Assign lots and collect spectra

Assign all preparations from one traceable parent lot to the same partition. Record the acquisition protocol and lot provenance. Vary sources and botanical origins across lots to match the intended screening population. Use opaque preparation IDs, keep test labels with the person who controls unblinding, and randomize acquisition order so pure and spiked preparations share the same measurement conditions. Reserve at least three development lots for grouped tuning and ensure both classes are represented in every inner training split. The calibration partition uses one pure preparation from each of its lots to choose a decision threshold. Multiple scans of one preparation are averaged.

Supply these inputs as UTF-8 CSV files.

| File | Required columns | Contents |
| --- | --- | --- |
| `partitions.csv` | `lot_id,partition` | Each parent lot once. Use `development`, `calibration` or `test`. |
| `fit-labels.csv` | `lot_id,preparation_id,dose_pct,adulterant` | Development and calibration preparation labels. Use dose `0` and adulterant `none` for pure honey. |
| `fit-spectra.csv` | `lot_id,preparation_id,instrument,scan_id` plus wavelength columns | Development and calibration scans. Use one file per instrument and wavelength headers such as `980.0nm`. |
| `test-spectra.csv` | Same schema as `fit-spectra.csv` | Blinded test scans. Assign unique scan IDs within each instrument. Keep the preparation ID constant across repeat scans and instruments. |
| `test-labels.csv` | Same schema as `fit-labels.csv` | The test custodian opens this file after the frozen scores have been recorded. |

Keep the wavelength window within measured coverage on every participating instrument. The default is the benchmark’s 780–1000 nm window. For instruments with different coverage, prespecify a common window using `--min-nm` and `--max-nm`. Keep both signal types and acquisition metadata documented. For a transfer experiment, scan the same test preparations on both instruments. The comparison applies the same fitted model and threshold to both readings. [Example paired honey spectroscopy study](https://doi.org/10.3390/s26092750)

## 3. Fit, freeze and score before unblinding

Example for one instrument called `screening-unit` and two syrups, `rice` and `sucrose`. Set the corresponding planning and fitting targets to the same values.

```powershell
python -m pip install -r requirements.txt
python followup.py freeze --partitions followup_data/partitions.csv --labels followup_data/fit-labels.csv --spectra followup_data/fit-spectra.csv --source-instrument screening-unit --adulterants rice sucrose --primary-model XGBoost --out followup_run/frozen.pkl
python followup.py score --frozen followup_run/frozen.pkl --spectra followup_data/test-spectra.csv --out followup_run/blind.json
```

The freeze step reuses the benchmark’s wavelength interpolation, SNV, training-only channel scaling, model grids and grouped tuning. For each model it chooses the lowest score threshold that attains the requested empirical specificity on the separate pure calibration lots. It stores the threshold, observed calibration specificity and number of calibration lots. Empirical calibration specificity based on few lots has substantial uncertainty. The blinded test gate uses independent lots and confidence bounds.

If you ran the [24-lot pilot](PILOT.md), add `--pilot-lots pilot_run/design/lot_inventory.csv` to `freeze`. The command checks for overlap with the final test partition using the stable parent lot IDs.

For two instruments, add `--target-instrument target-name`, keep the source name in `--source-instrument`, and pass each instrument file with its own `--spectra` argument. The target instrument supplies the calibration controls by default. Use `--calibration-instrument source-name` to evaluate direct transfer of a source-calibrated threshold. The `score` command checks that the test preparation IDs match across instruments. It accepts locally generated model bundles from the freeze step.

Record the SHA-256 printed by `score` in the study log before the test custodian opens the labels. The human-readable freeze receipt sits next to the model bundle as `frozen.json` and records the frozen rules, partitions, thresholds and input hashes.

## 4. Open labels and apply the decision rule

```powershell
$recordedDigest = 'paste-the-sha-256-recorded-before-unblinding'
python followup.py evaluate --receipt followup_run/frozen.json --scores followup_run/blind.json --expected-scores-sha256 $recordedDigest --labels followup_data/test-labels.csv --out followup_run/decision.json
```

The designated primary model passes only if all three one-sided 95% exact lower bounds meet their registered targets on the target instrument. The output includes lot counts, rates, bounds and per-syrup detection. For paired instruments it also reports target-versus-source changes and discordant lot counts. The output checks the recorded blind-score digest and verifies that the supplied receipt matches the configuration used to produce those scores. Evaluate each dataset into a fresh output path.

## Verification

Run `python -m unittest discover -s tests` to check lot separation, dose-panel completeness, calibration ties, sample-size calculations, paired instrument counts, and frozen-artifact integrity. Use `python followup_demo.py --out reproduced/followup_dry_run.json` to reproduce the [Auckland dry run](results/followup_dry_run.json) using the cached release. It holds out brands across stages and uses product–acquisition identifiers as lot surrogates for calibration and testing. The public archive lacks verified independent parent lots, so the dry-run intervals and decision serve only to verify the workflow.
