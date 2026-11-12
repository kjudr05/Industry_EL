import unittest
import pandas as pd
from src.validation import summarize_repeatability

class ValidationHelperTests(unittest.TestCase):
    def test_repeatability_summary_has_expected_metrics(self):
        data = pd.DataFrame({
            "quality_estimate": [80.0, 82.0, 81.0],
            "model_energy_estimate_kwh": [1.1, 1.2, 1.15],
            "line_energy_simulated_kwh": [0.8, 0.9, 0.85],
            "line_lead_time_min": [12.0, 13.0, 12.5],
        })
        result = summarize_repeatability(data)
        self.assertEqual(len(result), 4)
        self.assertIn("std_dev", result.columns)
        self.assertTrue((result["replicates"] == 3).all())

if __name__ == "__main__":
    unittest.main()
