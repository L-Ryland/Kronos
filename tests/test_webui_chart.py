import json
import os
import tempfile
import unittest

import pandas as pd

import webui.app as webui_app
from webui.app import app, create_prediction_chart


class CapturingPredictor:
    def __init__(self):
        self.calls = []

    def predict(
        self,
        df,
        x_timestamp,
        y_timestamp,
        pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
    ):
        self.calls.append(
            {
                "df": df.copy(),
                "x_timestamp": pd.Series(x_timestamp).copy(),
                "y_timestamp": pd.Series(y_timestamp).copy(),
                "pred_len": pred_len,
                "T": T,
                "top_p": top_p,
                "sample_count": sample_count,
            }
        )
        return pd.DataFrame(
            {
                "open": [100.0 + i for i in range(pred_len)],
                "high": [101.0 + i for i in range(pred_len)],
                "low": [99.0 + i for i in range(pred_len)],
                "close": [100.5 + i for i in range(pred_len)],
                "volume": [1000.0 + i for i in range(pred_len)],
            },
            index=pd.Series(y_timestamp),
        )


class WebUiChartTest(unittest.TestCase):
    def test_api_predict_without_start_date_forecasts_after_latest_candle(self):
        timestamps = pd.date_range("2026-01-01 00:00:00", periods=8, freq="h")
        df = pd.DataFrame(
            {
                "timestamps": timestamps,
                "open": [10, 11, 12, 13, 14, 15, 16, 17],
                "high": [11, 12, 13, 14, 15, 16, 17, 18],
                "low": [9, 10, 11, 12, 13, 14, 15, 16],
                "close": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5],
                "volume": [100] * 8,
            }
        )

        original_predictor = webui_app.predictor
        original_model_available = webui_app.MODEL_AVAILABLE
        original_save_prediction_results = webui_app.save_prediction_results
        fake_predictor = CapturingPredictor()

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
            df.to_csv(temp_file.name, index=False)
            temp_path = temp_file.name

        try:
            webui_app.predictor = fake_predictor
            webui_app.MODEL_AVAILABLE = True
            webui_app.save_prediction_results = lambda **_: None

            with app.test_client() as client:
                response = client.post(
                    "/api/predict",
                    json={
                        "file_path": temp_path,
                        "lookback": 4,
                        "pred_len": 2,
                        "temperature": 1.0,
                        "top_p": 0.9,
                        "sample_count": 1,
                    },
                )

            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertTrue(payload["success"])
            self.assertFalse(payload["has_comparison"])
            self.assertEqual(payload["actual_data"], [])

            call = fake_predictor.calls[0]
            self.assertEqual(
                call["x_timestamp"].tolist(),
                timestamps[-4:].to_series(index=range(4)).tolist(),
            )
            self.assertEqual(
                call["y_timestamp"].tolist(),
                [
                    timestamps[-1] + pd.Timedelta(hours=1),
                    timestamps[-1] + pd.Timedelta(hours=2),
                ],
            )

            chart = json.loads(payload["chart"])
            traces = {trace["name"]: trace for trace in chart["data"]}
            self.assertEqual(
                traces["Historical Kline"]["x"][-1],
                timestamps[-1].isoformat(),
            )
            self.assertEqual(
                traces["Prediction Kline"]["x"][0],
                (timestamps[-1] + pd.Timedelta(hours=1)).isoformat(),
            )
            self.assertNotIn("Actual Kline", traces)
        finally:
            webui_app.predictor = original_predictor
            webui_app.MODEL_AVAILABLE = original_model_available
            webui_app.save_prediction_results = original_save_prediction_results
            os.unlink(temp_path)

    def test_prediction_chart_draws_kline_candlesticks(self):
        df = pd.DataFrame(
            {
                "timestamps": pd.date_range("2026-01-01", periods=6, freq="h"),
                "open": [10, 11, 12, 13, 14, 15],
                "high": [11, 12, 13, 14, 15, 16],
                "low": [9, 10, 11, 12, 13, 14],
                "close": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
                "volume": [100] * 6,
            }
        )
        pred_df = pd.DataFrame(
            {
                "open": [13.5, 14.0],
                "high": [14.5, 15.0],
                "low": [13.0, 13.8],
                "close": [14.0, 14.5],
                "volume": [100, 100],
            }
        )
        actual_df = df.iloc[4:6]

        chart = json.loads(
            create_prediction_chart(
                df, pred_df, lookback=4, pred_len=2, actual_df=actual_df
            )
        )

        candlestick_traces = {
            trace["name"]: trace
            for trace in chart["data"]
            if trace.get("type") == "candlestick"
        }
        line_traces = [trace for trace in chart["data"] if trace.get("type") == "scatter"]

        self.assertEqual(
            set(candlestick_traces),
            {"Historical Kline", "Prediction Kline", "Actual Kline"},
        )
        self.assertEqual(candlestick_traces["Prediction Kline"]["open"], [13.5, 14.0])
        self.assertEqual(candlestick_traces["Prediction Kline"]["high"], [14.5, 15.0])
        self.assertEqual(candlestick_traces["Prediction Kline"]["low"], [13.0, 13.8])
        self.assertEqual(candlestick_traces["Prediction Kline"]["close"], [14.0, 14.5])
        self.assertEqual(line_traces, [])

    def test_webui_serves_plotly_from_local_app(self):
        with app.test_client() as client:
            index_response = client.get("/")
            plotly_response = client.get("/vendor/plotly.min.js")

            self.assertEqual(index_response.status_code, 200)
            self.assertIn(b'src="/vendor/plotly.min.js"', index_response.data)
            self.assertNotIn(b"cdn.plot.ly", index_response.data)
            self.assertNotIn(b"start_date: startDate", index_response.data)
            self.assertEqual(plotly_response.status_code, 200)
            self.assertIn(b"Plotly", plotly_response.data[:100_000])
            plotly_response.close()


if __name__ == "__main__":
    unittest.main()
