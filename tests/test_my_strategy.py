import pandas as pd
import unittest

from my_strategy import calculate_backtest_metrics, run_kronos_backtest


class FakePredictor:
    def __init__(self, expected_returns):
        self.expected_returns = list(expected_returns)
        self.calls = 0

    def predict(self, df, x_timestamp, y_timestamp, pred_len, **kwargs):
        last_close = df["close"].iloc[-1]
        expected_return = self.expected_returns[self.calls]
        self.calls += 1
        return pd.DataFrame(
            {
                "open": [last_close] * pred_len,
                "high": [last_close] * pred_len,
                "low": [last_close] * pred_len,
                "close": [last_close * (1 + expected_return)] * pred_len,
                "volume": [0.0] * pred_len,
                "amount": [0.0] * pred_len,
            },
            index=y_timestamp,
        )


def make_ohlcv(rows):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="4h"),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "close": [101.0, 100.0, 103.0, 102.0, 106.0, 104.0, 108.0],
            "volume": [10.0] * rows,
        }
    )


class MyStrategyTest(unittest.TestCase):
    def test_run_kronos_backtest_converts_predictions_to_positions_and_equity(self):
        predictor = FakePredictor([0.02, -0.02, 0.0])

        result = run_kronos_backtest(
            make_ohlcv(7),
            predictor,
            lookback=2,
            pred_len=1,
            threshold=0.01,
            fee=0.001,
            allow_short=True,
            initial_capital=10_000,
        )

        self.assertEqual(result["position"].tolist(), [1, -1, 0])
        self.assertEqual(result["expected_return"].round(4).tolist(), [0.02, -0.02, 0.0])
        self.assertGreater(result["equity"].iloc[-1], 10_000)

    def test_calculate_backtest_metrics_summarizes_result_frame(self):
        bt = pd.DataFrame(
            {
                "position": [0, 1, 1, 0],
                "net_return": [0.0, 0.02, -0.01, 0.01],
                "equity": [10_000.0, 10_200.0, 10_098.0, 10_198.98],
                "drawdown": [0.0, 0.0, -0.01, 0.0],
            }
        )

        metrics = calculate_backtest_metrics(bt, initial_capital=10_000)

        self.assertAlmostEqual(metrics["total_return"], 0.019898)
        self.assertEqual(metrics["max_drawdown"], -0.01)
        self.assertEqual(metrics["win_rate"], 0.5)
        self.assertEqual(metrics["trades"], 2)


if __name__ == "__main__":
    unittest.main()
