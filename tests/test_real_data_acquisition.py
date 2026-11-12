import unittest
from src.real_data_acquisition import load_catalog, inventory_files

class TestRealDataAcquisition(unittest.TestCase):
    def test_catalog_has_three_distinct_sources(self):
        catalog = load_catalog()
        self.assertEqual(catalog["phase"], "12A")
        self.assertGreaterEqual(len(catalog["datasets"]), 3)
        ids = [d["dataset_id"] for d in catalog["datasets"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_inventory_schema(self):
        df = inventory_files()
        self.assertEqual(list(df.columns), ["file", "bytes", "rows", "columns", "status"])

if __name__ == "__main__":
    unittest.main()
