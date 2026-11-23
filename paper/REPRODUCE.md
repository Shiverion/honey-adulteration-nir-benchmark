# Reproduce the honey adulteration study

## Run the analysis

From the archive root, use Python 3.13 and install the versions in `requirements.txt`. The study uses a single CPU thread. The recorded PyTorch build also supports GPU execution, but these scripts use the CPU.

```powershell
python -m pip install -r requirements.txt
python benchmark.py --download --out reproduced/benchmark
python summarize.py --out reproduced/benchmark
python padua_transfer.py --download --out reproduced/padua_transfer
python paper/verify_analysis.py --benchmark reproduced/benchmark --transfer reproduced/padua_transfer
```

The baseline benchmark took approximately eight minutes on the study host. The transfer command refits both Auckland wavelength windows, writes the matched-window frozen model, verifies the Padua download and scores its 80 FOSS spectra. Use a fresh output directory for each transfer run. With all three files cached in `data/`, both pipelines run offline and check the same hashes. Time and serialization checksums can vary by environment. Verify prediction and metric values when comparing runs.

| Source | Filename expected in `data/` | Role |
| --- | --- | --- |
| Auckland Figshare v1 [source record](https://doi.org/10.17608/k6.auckland.16441686.v1) | `adulteration_dataset_26_08_2021.csv` | Nested benchmark and development of both transfer studies |
| NIRPY Research [pinned CSV](https://github.com/nevernervous78/nirpyresearch/blob/0efbcf0fcd97580dbffd79e857d70d8f001cf5ff/data/Honey2021.csv) | `Honey2021.csv` | Labeled Argentine preparations |
| University of Padua [source record](https://doi.org/10.25430/researchdata.cab.unipd.it.00001770) | `Padua_Dataset_Honey.xlsx` | Padua FOSS flag-rate study |

The download URLs and SHA-256 digests are in `datasets.py`. The archive references the upstream files under their respective licenses. Its `results/` folder contains derived predictions and study records for inspection.

## Read the outputs

- `results/metrics.csv`, `by_brand.csv`, `by_dose.csv`, `robustness.csv` and `efficiency.csv` contain the seven-model results. `comparisons.csv` reports paired effects, nonzero brand contributions, and minimum attainable one-sided p-values. `brand_uncertainty.csv` contains the model-level brand bootstrap intervals.
- `results/predictions.csv`, `splits.json`, `tuning.csv` and `timings.csv` provide the underlying records. The two observation manifests identify the source records used and the numbers of input scans aggregated per observation.
- `results/padua_transfer/auckland_metrics.csv` and `paired_differences.csv` report the two-window analysis. `padua_flag_rates.csv` contains the target counts for both signal representations. `feature_support.csv` counts, by observation, features after SNV that fall outside their Auckland development minima or maxima. The prediction files retain scores, thresholds and decisions. `split_manifest.json` records the development, calibration and test brand roles.
- `paper/manuscript.md` is the manuscript source. `paper/figures/` and `results/padua_transfer/` contain the publication figures as vector PDFs and raster previews. `python paper/figures.py` rebuilds the baseline figure from the supplied results. `python paper/render_manuscript.py` builds the editable manuscript using `python-docx` 1.2.0.

## Submission handoff

The Word and PDF manuscript files are anonymous for review. Supply the author order, affiliations and corresponding-author details through the chosen journal or preprint service. Confirm authorship contributions, funding, conflicts, data and code sharing permissions, and the venue’s disclosure requirements. The work is a retrospective public-data study. The matched-window results and manuscript do not incorporate any newly collected honey lots.
