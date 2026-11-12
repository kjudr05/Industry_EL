import tempfile
import unittest
from src.simulator import generate_batches
from src.deep_models import train_ann_models

class DeepModelTests(unittest.TestCase):
    def test_ann_training_returns_regression_and_classification_metrics(self):
        data = generate_batches(60, 202)
        with tempfile.TemporaryDirectory() as out:
            metrics, models = train_ann_models(data, output_dir=out, random_state=202)
        self.assertIn("ANN Regressor", models)
        self.assertIn("ANN Classifier", models)
        self.assertIn("MAE", metrics["regression"][0])
        self.assertIn("F1_macro", metrics["classification"][0])

if __name__ == "__main__":
    unittest.main()
