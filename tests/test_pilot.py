from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import pandas as pd

from followup import freeze
from pilot import build_folds, inspect_scan, make_plan, paired_repeats, spectral_repeatability


class PilotTests(TestCase):
    def test_design_has_complete_panels_and_opaque_randomized_scan_queue(self):
        lots = [f"L{i:02d}" for i in range(24)]
        labels, schedule = make_plan(lots, ["rice", "sucrose"], "unit", 6, 17)
        self.assertEqual(len(labels), 120)
        self.assertEqual(labels["preparation_id"].nunique(), 120)
        self.assertTrue(labels.groupby("lot_id").size().eq(5).all())
        self.assertEqual(schedule.groupby("day").size().to_dict(), {"day_1": 120, "day_2": 30})
        day_1 = schedule.loc[schedule["day"].eq("day_1")]
        day_2 = schedule.loc[schedule["day"].eq("day_2")]
        self.assertTrue(day_2.groupby("lot_id").size().eq(5).all())
        self.assertEqual(day_2["lot_id"].nunique(), 6)
        self.assertTrue(set(day_2["preparation_id"]) <= set(day_1["preparation_id"]))
        self.assertFalse({"dose_pct", "adulterant"} & set(schedule))
        self.assertFalse(schedule["scan_id"].duplicated().any())
        self.assertNotEqual(day_1["lot_id"].tolist(), sorted(day_1["lot_id"].tolist()))

    def test_each_lot_is_held_out_once_without_phase_overlap(self):
        lots = np.repeat([f"L{i:02d}" for i in range(24)], 5)
        tested = np.zeros(len(lots), dtype=int)
        for dev, cal, test in build_folds(lots, 6, 10, 17):
            self.assertTrue(np.all(dev.astype(int) + cal + test == 1))
            self.assertEqual((len(set(lots[dev])), len(set(lots[cal])), len(set(lots[test]))), (10, 10, 4))
            self.assertFalse(set(lots[dev]) & set(lots[cal]))
            self.assertFalse(set(lots[dev]) & set(lots[test]))
            self.assertFalse(set(lots[cal]) & set(lots[test]))
            tested += test
        self.assertTrue(np.all(tested == 1))

    def test_single_export_reports_partial_common_window(self):
        meta = pd.DataFrame({"scan_id": ["s1"], "instrument": ["unit"]})
        result = inspect_scan(meta, np.array([[0.2, 0.3, 0.5]]), np.array([910.0, 1000.0, 1300.0]))
        self.assertFalse(result["covers_780_1000_nm"])
        self.assertEqual(result["native_channels"], 3)

    def test_repeat_stability_pairs_the_same_preparation_and_fold(self):
        baseline = pd.DataFrame({"lot_id": ["A", "A"], "preparation_id": ["P1", "P2"], "fold": [1, 1],
                                 "score": [-0.2, 0.5], "prediction": [0, 1]})
        repeat = baseline.iloc[1:].assign(score=-0.1, prediction=0)
        result = paired_repeats(baseline, repeat)
        self.assertEqual(result["decision_flips"], 1)
        self.assertAlmostEqual(result["median_absolute_score_change"], 0.6)
        with self.assertRaisesRegex(ValueError, "same fold"):
            paired_repeats(baseline, repeat.assign(fold=2))

    def test_spectral_repeatability_removes_gain_and_offset(self):
        grid = np.array([800, 900, 1000])
        baseline = pd.DataFrame({"preparation_id": ["P1", "P2"], 0: [1.0, 2.0], 1: [3.0, 4.0], 2: [2.0, 7.0]})
        repeat = baseline.iloc[[1, 0]].copy()
        repeat[[0, 1, 2]] = repeat[[0, 1, 2]] * 2 + 5
        self.assertAlmostEqual(spectral_repeatability(baseline, repeat, grid)["p95_snv_rms"], 0)

    def test_freeze_excludes_pilot_lots_from_the_final_test(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            partitions = base / "partitions.csv"
            partitions.write_text("lot_id,partition\nA,development\nB,calibration\nC,test\n")
            pilot_lots = base / "pilot.csv"
            pilot_lots.write_text("lot_id\nC\n")
            args = Namespace(out=base / "frozen.pkl", adulterants=["rice"], min_nm=780, max_nm=1000,
                             source_instrument="unit", target_instrument=None, calibration_instrument=None,
                             partitions=partitions, pilot_lots=pilot_lots)
            with self.assertRaisesRegex(ValueError, "Pilot lots cannot appear"):
                freeze(args)
