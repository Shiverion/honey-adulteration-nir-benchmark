import json
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from followup import MODELS, evaluate, json_bytes, load_partitions, sha256
from followup_stats import complete_panel, rate_summary, sample_size_plan, threshold_for_specificity


class FollowupTests(TestCase):
    def test_one_threshold_handles_tied_pure_scores_conservatively(self):
        scores = np.array([0.1, 0.4, 0.4, 0.9])
        threshold = threshold_for_specificity(scores, 0.5)
        self.assertEqual(threshold, 0.4)
        self.assertEqual(np.count_nonzero(scores <= threshold), 3)

    def test_constant_lot_size_and_exact_gate(self):
        targets, alternatives = (0.90, 0.80, 0.80), (0.96, 0.90, 0.92)
        design = sample_size_plan(targets, alternatives, 0.05, 0.80, 5000)
        n = design["test_lots"]
        self.assertGreaterEqual(design["joint_power_lower_bound"], 0.80)
        self.assertEqual(design["endpoints"]["specificity"]["first_lot_count_where_passing_is_possible"], 29)
        for endpoint in design["endpoints"].values():
            k = endpoint["minimum_successes_at_planned_size"]
            self.assertLessEqual(binomtest(k, n, endpoint["target"], alternative="greater").pvalue, 0.05)
            self.assertGreater(binomtest(k - 1, n, endpoint["target"], alternative="greater").pvalue, 0.05)
        self.assertFalse(rate_summary(np.ones(28), 0.90, 0.05)["meets_target"])
        self.assertTrue(rate_summary(np.ones(29), 0.90, 0.05)["meets_target"])

    def test_partitions_refuse_a_shared_lot(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "partitions.csv"
            path.write_text("lot_id,partition\nA,development\nB,calibration\nA,test\n")
            with self.assertRaisesRegex(ValueError, "Each lot must appear once"):
                load_partitions(path)

    def test_test_panel_requires_every_syrup_at_each_dose(self):
        incomplete = pd.DataFrame({"lot_id": ["A", "A", "A"], "dose_pct": [0, 5, 10],
                                   "adulterant": ["none", "rice", "glucose"]})
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            complete_panel(incomplete, ["rice", "glucose"])

    def test_evaluation_preserves_pairing_and_checks_the_blind(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            receipt = {
                "frozen_sha256": "example", "primary_model": "XGBoost", "alpha": 0.05,
                "specificity": 0.9, "sensitivity_5pct": 0.8, "sensitivity_10pct": 0.8,
                "adulterants": ["rice"], "source_instrument": "source", "target_instrument": "target",
                "partitions": {"test": ["A", "B"]}, "models": {model: {"threshold": 0} for model in MODELS},
            }
            labels = [(lot, f"{lot}{dose}", dose, "rice" if dose else "none")
                      for lot in ("A", "B") for dose in (0, 5, 10)]
            raw_scores = []
            for model in MODELS:
                for instrument in ("source", "target"):
                    for lot, prep, dose, _ in labels:
                        decision = int(dose > 0 and (instrument == "source" or lot == "A"))
                        raw_scores.append({"instrument": instrument, "model": model, "lot_id": lot,
                                           "preparation_id": prep, "prediction": decision, "score": decision * 2 - 1})
            args = Namespace(receipt=base / "receipt.json", scores=base / "scores.json",
                             labels=base / "labels.csv", out=base / "decision.json")
            args.receipt.write_bytes(json_bytes(receipt))
            encoded = json_bytes({"frozen_receipt_sha256": sha256(json_bytes(receipt)), "predictions": raw_scores})
            args.scores.write_bytes(encoded)
            args.expected_scores_sha256 = sha256(encoded)
            pd.DataFrame(labels, columns=["lot_id", "preparation_id", "dose_pct", "adulterant"]).to_csv(args.labels, index=False)
            evaluate(args)
            result = json.loads(args.out.read_bytes())
            self.assertEqual(result["go_decision"], "fail")
            self.assertEqual(result["paired_instrument_changes"]["XGBoost"]["sensitivity_5pct"],
                             {"target_minus_source": -0.5, "target_only": 0, "source_only": 1})
            args.scores.write_bytes(encoded + b" ")
            with self.assertRaisesRegex(ValueError, "recorded before unblinding"):
                evaluate(args)
            args.scores.write_bytes(encoded)
            args.receipt.write_bytes(json_bytes({**receipt, "primary_model": "MLP"}))
            with self.assertRaisesRegex(ValueError, "configuration used to score"):
                evaluate(args)
