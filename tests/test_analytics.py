"""Quick smoke tests for the NEXUS analytics layer.

Run from the project root with: python -m unittest discover -s tests
"""
import unittest
import pandas as pd

from nexus.analytics import (
    run_arima,
    run_association_rules,
    run_classification,
    run_clustering,
    run_regression,
)
from nexus.data import add_derived_fields, make_demo_data


class AnalyticsSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sales, cls.transactions = make_demo_data(days=210)
        cls.sales = add_derived_fields(cls.sales)

    def test_regression(self):
        result = run_regression(self.sales, "sales_units", ["marketing_spend", "price", "discount_pct", "product", "region"])
        self.assertIn("r2", result)
        self.assertGreater(result["test_rows"], 0)

    def test_classification(self):
        result = run_classification(self.sales, "stockout_risk", ["sales_units", "inventory_on_hand", "lead_time_days", "marketing_spend"])
        self.assertIn("accuracy", result)
        self.assertGreaterEqual(result["accuracy"], 0)

    def test_clustering(self):
        result = run_clustering(self.sales, ["sales_units", "marketing_spend", "price", "inventory_on_hand"], 3)
        self.assertEqual(result["plot"]["cluster"].nunique(), 3)

    def test_association_rules(self):
        result = run_association_rules(self.transactions, "transaction_id", "product", .02, .2, .9)
        self.assertIsInstance(result, pd.DataFrame)

    def test_arima(self):
        weekly = self.sales.groupby(pd.Grouper(key="date", freq="W-SUN"))["sales_units"].sum()
        result = run_arima(weekly, 3, (1, 1, 1))
        self.assertEqual(len(result["forecast"]), 3)


if __name__ == "__main__":
    unittest.main()
