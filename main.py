import os
import math
import json
import random
import datetime
import concurrent.futures
import numpy as np
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Load credentials securely from environment variables
ALPACA_API_KEY = os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")

trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

TARGET_UNIVERSE = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "NFLX"]

class ChartScannerSkill:
    @staticmethod
    def run(symbol: str, df: pd.DataFrame):
        if df is None or len(df) < 20:
            return {"signal": "INVALID_DATA", "current_price": 0, "volume": 0}
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        vol_sma = df['volume'].rolling(20).mean().iloc[-1]
        
        breakout = latest['close'] > prev['high'] and latest['close'] > sma_20
        volume_surge = latest['volume'] > (vol_sma * 1.1)
        
        signal = "BULLISH_BREAKOUT" if (breakout and volume_surge) else "WATCH"
        return {
            "signal": signal,
            "current_price": float(latest['close']),
            "volume": float(latest['volume']),
            "vol_avg": float(vol_sma)
        }

class EntryTimerSkill:
    @staticmethod
    def run(current_price: float, df: pd.DataFrame):
        atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1] if len(df) >= 14 else current_price * 0.02
        return {"entry_price": round(current_price, 2), "atr_14": round(float(atr), 2)}

class ExitPlannerSkill:
    @staticmethod
    def run(entry_price: float, atr: float):
        stop_loss = round(entry_price - (1.5 * atr), 2)
        take_profit = round(entry_price + (3.0 * atr), 2)
        return {"stop_loss": stop_loss, "take_profit": take_profit}

class TimeMachineSkill:
    @staticmethod
    def run(entry: float, stop: float, target: float, df: pd.DataFrame, num_simulations: int = 2000):
        returns = df['close'].pct_change().dropna().values
        daily_vol = np.std(returns) if len(returns) > 0 else 0.02
        hits_target = 0
        
        for _ in range(num_simulations):
            price = entry
            for _ in range(15):
                price *= (1 + np.random.normal(0, daily_vol))
                if price >= target:
                    hits_target += 1
                    break
                elif price <= stop:
                    break
        win_rate = round((hits_target / num_simulations) * 100, 2)
        return {"stat_win_rate": f"{win_rate}%"}

class RiskManagerSkill:
    @staticmethod
    def run(entry: float, stop: float, buying_power: float):
        risk_per_share = max(0.01, entry - stop)
        qty = math.floor((buying_power * 0.01) / risk_per_share)
        max_qty = math.floor((buying_power * 0.1) / entry)
        return {"recommended_qty": max(1, min(qty, max_qty))}

class ExecutionAgentSkill:
    @staticmethod
    def run(symbol: str, qty: int, side: OrderSide):
        try:
            req = MarketOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY)
            order = trading_client.submit_order(order_data=req)
            return {"status": "SUCCESS", "order_id": str(order.id)}
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}

class JarvisMarketScanner:
    def __init__(self, symbols):
        self.symbols = symbols

    def fetch_data(self, symbol):
        try:
            start_date = datetime.datetime.now() - datetime.timedelta(days=45)
            bars = data_client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start_date
            ))
            df = bars.df
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(symbol)
            return symbol, df
        except Exception:
            return symbol, None

    def scan_and_trade(self):
        print(f"🔍 J.A.R.V.I.S. // RUNNING MARKET SCAN AT {datetime.datetime.now()}...")
        try:
            account = trading_client.get_account()
            buying_power = float(account.buying_power)

            qualified_setups = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_symbol = {executor.submit(self.fetch_data, sym): sym for sym in self.symbols}
                for future in concurrent.futures.as_completed(future_to_symbol):
                    sym, df = future.result()
                    if df is not None and not df.empty:
                        s1 = ChartScannerSkill.run(sym, df)
                        if s1["current_price"] > 0:
                            s3 = EntryTimerSkill.run(s1["current_price"], df)
                            s4 = ExitPlannerSkill.run(s3["entry_price"], s3["atr_14"])
                            s5 = TimeMachineSkill.run(s3["entry_price"], s4["stop_loss"], s4["take_profit"], df)
                            s6 = RiskManagerSkill.run(s3["entry_price"], s4["stop_loss"], buying_power)
                            
                            qualified_setups.append({
                                "symbol": sym,
                                "price": s1["current_price"],
                                "signal": s1["signal"],
                                "win_rate": s5["stat_win_rate"],
                                "qty": s6["recommended_qty"]
                            })

            qualified_setups.sort(key=lambda x: float(x["win_rate"].replace("%", "")), reverse=True)

            if qualified_setups:
                top_pick = qualified_setups[0]
                print(f"⚡ BEST SETUP FOUND: {top_pick['symbol']} | Win Rate: {top_pick['win_rate']} | Action: BUY {top_pick['qty']} shares")
                exec_res = ExecutionAgentSkill.run(top_pick['symbol'], top_pick['qty'], OrderSide.BUY)
                print(f"📋 EXECUTION RESULT: {json.dumps(exec_res)}")
            else:
                print("⏳ No optimal setups found in this scan cycle.")
                
        except Exception as e:
            print(f"⚠️ Scan cycle error: {str(e)}")

if __name__ == "__main__":
    scanner = JarvisMarketScanner(TARGET_UNIVERSE)
    scanner.scan_and_trade()
