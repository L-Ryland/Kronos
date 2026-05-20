from __future__ import annotations

from typing import Any

import pandas as pd


REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close"]
FEATURE_COLUMNS = ["open", "high", "low", "close", "volume"]


def run_kronos_backtest(
    df: pd.DataFrame,
    predictor: Any,
    lookback: int = 400,
    pred_len: int = 6,
    threshold: float = 0.005,
    fee: float = 0.0006,
    allow_short: bool = True,
    initial_capital: float = 10_000,
) -> pd.DataFrame:
    """Run a walk-forward backtest using a Kronos predictor.

    The predictor is expected to expose the same API as KronosPredictor.predict.
    Signals are generated from the forecasted close:
    long when expected return > threshold, short when below -threshold, otherwise flat.
    """
    _validate_inputs(df, lookback, pred_len, threshold, fee, initial_capital)

    market_df = df.sort_values("timestamp").reset_index(drop=True).copy()
    if "volume" not in market_df.columns:
        market_df["volume"] = 0.0

    equity = float(initial_capital)
    previous_position = 0
    rows = []

    for i in range(lookback, len(market_df) - pred_len - 1):
        x_df = market_df.iloc[i - lookback : i][FEATURE_COLUMNS].reset_index(drop=True)
        x_timestamp = market_df.iloc[i - lookback : i]["timestamp"].reset_index(drop=True)
        y_timestamp = market_df.iloc[i : i + pred_len]["timestamp"].reset_index(drop=True)

        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=False,
        )

        current_close = market_df.loc[i - 1, "close"]
        predicted_close = pred_df["close"].iloc[-1]
        expected_return = predicted_close / current_close - 1
        position = _signal_to_position(expected_return, threshold, allow_short)

        entry = market_df.loc[i, "open"]
        exit_price = market_df.loc[i, "close"]
        gross_return = position * (exit_price / entry - 1)
        trade_cost = fee * abs(position - previous_position)
        net_return = gross_return - trade_cost

        equity *= 1 + net_return
        previous_position = position

        rows.append(
            {
                "timestamp": market_df.loc[i, "timestamp"],
                "entry": entry,
                "exit": exit_price,
                "predicted_close": predicted_close,
                "expected_return": expected_return,
                "position": position,
                "gross_return": gross_return,
                "trade_cost": trade_cost,
                "net_return": net_return,
                "equity": equity,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result["drawdown"] = result["equity"] / result["equity"].cummax() - 1
    result.attrs["initial_capital"] = float(initial_capital)
    return result


def calculate_backtest_metrics(
    backtest: pd.DataFrame,
    initial_capital: float | None = None,
) -> dict[str, float]:
    """Summarize a DataFrame returned by run_kronos_backtest."""
    if backtest.empty:
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "trades": 0.0,
            "final_equity": float(initial_capital or 0.0),
        }

    if initial_capital is None:
        initial_capital = backtest.attrs.get("initial_capital", backtest["equity"].iloc[0])

    position_changes = backtest["position"].diff().fillna(backtest["position"])
    trades = int((position_changes != 0).sum())
    active_returns = backtest.loc[backtest["position"] != 0, "net_return"]
    win_rate = float((active_returns > 0).mean()) if len(active_returns) else 0.0
    final_equity = float(backtest["equity"].iloc[-1])

    return {
        "total_return": final_equity / float(initial_capital) - 1,
        "max_drawdown": float(backtest["drawdown"].min()),
        "win_rate": win_rate,
        "trades": float(trades),
        "final_equity": final_equity,
    }


def _signal_to_position(
    expected_return: float,
    threshold: float,
    allow_short: bool,
) -> int:
    if expected_return > threshold:
        return 1
    if allow_short and expected_return < -threshold:
        return -1
    return 0


def _validate_inputs(
    df: pd.DataFrame,
    lookback: int,
    pred_len: int,
    threshold: float,
    fee: float,
    initial_capital: float,
) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if lookback <= 0:
        raise ValueError("lookback must be greater than 0.")
    if pred_len <= 0:
        raise ValueError("pred_len must be greater than 0.")
    if threshold < 0:
        raise ValueError("threshold cannot be negative.")
    if fee < 0:
        raise ValueError("fee cannot be negative.")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be greater than 0.")
    if len(df) <= lookback + pred_len + 1:
        raise ValueError(
            "Not enough rows for backtesting. Need more than "
            "lookback + pred_len + 1 rows."
        )
