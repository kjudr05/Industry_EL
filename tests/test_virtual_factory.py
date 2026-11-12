import unittest

from src.simulator import generate_batches, PRODUCTS
from src.production_line import simulate_production

class VirtualChipFactoryTests(unittest.TestCase):
    def test_synthetic_generator_covers_three_products(self):
        df = generate_batches(n_per_product=10, seed=7)
        self.assertEqual(len(df), 30)
        self.assertEqual(set(df["product_type"].unique()), set(PRODUCTS))
        self.assertTrue(df["quality_score"].between(0, 100).all())

    def test_dynamic_settings_change_thermal_stage_duration(self):
        base_events, _, _ = simulate_production(
            products=["Fried Potato Chips"], batches_per_product=1, seed=17,
            process_settings={"process_temperature_c": 175, "process_time_sec": 150,
                              "slice_thickness_mm": 1.2, "pretreatment_time_min": 5,
                              "line_speed_m_per_s": 1.0},
        )
        altered_events, _, altered_metrics = simulate_production(
            products=["Fried Potato Chips"], batches_per_product=1, seed=17,
            process_settings={"process_temperature_c": 175, "process_time_sec": 200,
                              "slice_thickness_mm": 1.2, "pretreatment_time_min": 5,
                              "line_speed_m_per_s": 1.0},
        )
        base_frying = base_events.loc[base_events["stage"] == "Frying", "process_time_min"].iloc[0]
        altered_frying = altered_events.loc[altered_events["stage"] == "Frying", "process_time_min"].iloc[0]
        self.assertNotEqual(base_frying, altered_frying)
        self.assertTrue(altered_metrics["kpis"]["dynamic_settings_applied"])

    def test_line_simulation_completes_batches(self):
        _, batches, metrics = simulate_production(
            products=PRODUCTS, batches_per_product=2, seed=7, machine_capacity=1
        )
        self.assertEqual(len(batches), 6)
        self.assertTrue((batches["status"] == "Completed").all())
        self.assertEqual(metrics["kpis"]["batches_completed"], 6)


    def test_classifier_accepts_string_quality_labels(self):
        import tempfile
        from src.ml_models import train_models
        df = generate_batches(40, 23)
        with tempfile.TemporaryDirectory() as output_dir:
            metrics, fitted, test_data = train_models(df, output_dir=output_dir, random_state=23)
        self.assertEqual(len(metrics["classification"]), 2)
        self.assertTrue(all(0.0 <= row["Accuracy"] <= 1.0 for row in metrics["classification"]))
        self.assertTrue(set(test_data["y_test_cls"]).issubset({"Pass", "Review"}))


if __name__ == "__main__":
    unittest.main()
