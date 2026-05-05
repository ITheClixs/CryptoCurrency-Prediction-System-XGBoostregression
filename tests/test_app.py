import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

import src.app as app_module


class DummyPredictor:
    def get_data(self, ticker, days):
        index = pd.date_range("2026-01-01", periods=max(days, 1))
        return pd.DataFrame(
            {"Close": [100.0 + i for i in range(len(index))]},
            index=index,
        )

    def get_current_price(self, crypto):
        return 100.0

    def predict_price(self, crypto, days=7):
        return 125.0


class FallbackAwarePredictor:
    def get_data(self, ticker, days):
        raise ValueError("primary data source unavailable")

    def get_current_price(self, crypto):
        return 100.0

    def predict_price(self, crypto, days=7):
        return 125.0


class AppRenderingTests(unittest.TestCase):
    def setUp(self):
        self.original_predictor = app_module.predictor
        app_module.predictor = DummyPredictor()
        app_module.app.config.update(TESTING=True)
        self.client = app_module.app.test_client()

    def tearDown(self):
        app_module.predictor = self.original_predictor

    def test_template_renders_chart_values_from_dictionary_key(self):
        with app_module.app.app_context():
            html = app_module.app.jinja_env.get_template("index.html").render(
                prediction=None,
                chart_data={
                    "labels": ["2026-01-01"],
                    "values": [123.45],
                    "crypto_symbol": "BTC",
                },
                crypto_symbol="BTC",
                days_to_predict=7,
            )

        self.assertIn("123.45", html)

    def test_post_with_invalid_days_returns_error_page_not_500(self):
        response = self.client.post("/", data={"crypto": "BTC", "days": "abc"})

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Days must be an integer", response.data)

    def test_post_clamps_days_to_supported_range(self):
        response = self.client.post("/", data={"crypto": "BTC", "days": "999"})

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Predicted in 90 days", response.data)

    def test_post_uses_fallback_aware_current_price_path(self):
        app_module.predictor = FallbackAwarePredictor()

        response = self.client.post("/", data={"crypto": "BTC", "days": "7"})

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Current Price: <strong>$100.00</strong>", response.data)
        self.assertIn(b"Predicted in 7 days", response.data)


class MarketDataNormalizationTests(unittest.TestCase):
    def test_get_data_flattens_single_ticker_yfinance_multiindex_columns(self):
        index = pd.date_range("2026-01-01", periods=3)
        columns = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Volume"], ["BTC-USD"]],
            names=["Price", "Ticker"],
        )
        raw_data = pd.DataFrame(
            [
                [100, 101, 99, 100.5, 1000],
                [101, 102, 100, 101.5, 1100],
                [102, 103, 101, 102.5, 1200],
            ],
            index=index,
            columns=columns,
        )

        with patch("yfinance.download", return_value=raw_data):
            data = app_module.CryptoPredictor().get_data("BTC-USD", 3)

        self.assertEqual(["Open", "High", "Low", "Close", "Volume"], list(data.columns))
        self.assertIsInstance(data["Close"], pd.Series)


class ModelArtifactTests(unittest.TestCase):
    def test_invalid_pickled_artifact_is_ignored(self):
        with TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "crypto_predictor.pkl"
            with artifact_path.open("wb") as artifact_file:
                import pickle

                pickle.dump(np.array([1, 2, 3]), artifact_file)

            predictor = app_module.CryptoPredictor()
            loaded = app_module.load_model_artifact(predictor, str(artifact_path))

        self.assertFalse(loaded)
        self.assertIsNone(predictor.model)
        self.assertIsNone(predictor.scaler)
        self.assertFalse(predictor.is_trained)


if __name__ == "__main__":
    unittest.main()
