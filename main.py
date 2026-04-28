import os
import queue
import threading
import time
from datetime import datetime, timedelta

import pandas as pd
import pyupbit

try:
    from rich import box
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


TICKERS = ["KRW-BTC", "KRW-ETH", "KRW-SOL", "KRW-XRP"]
INTERVAL = "minute5"
CANDLE_COUNT = 250
BACKTEST_CANDLE_COUNT = 600
OPTIMIZER_DAYS = 30
OPTIMIZER_CANDLE_COUNT = 9_500

STRATEGY_NAME = "5min Volume Breakout + Short Trend Following"
START_CASH = 1_000_000
TRADE_RATIO = 0.25
MAX_POSITIONS = 2
DAILY_MAX_LOSS = -0.03
REENTRY_COOLDOWN_CANDLES = 3
FEE_RATE = 0.0005

TAKE_PROFIT = 0.012
STOP_LOSS = -0.008
TRAILING_STOP = -0.006

LOOP_SECONDS = 60
TRADES_FILE = "trades.csv"

console = Console() if RICH_AVAILABLE else None


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_krw(value):
    return f"{value:,.0f}"


def format_percent(value):
    return f"{value:.2f}%"


def print_line(message):
    if RICH_AVAILABLE:
        console.print(message)
    else:
        print(message)


def get_ohlcv_with_timeout(ticker, count, to, timeout=10):
    result_queue = queue.Queue(maxsize=1)

    def worker():
        try:
            df = pyupbit.get_ohlcv(ticker, interval=INTERVAL, count=count, to=to)
            result_queue.put(("ok", df))
        except Exception as e:
            result_queue.put(("error", e))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    try:
        status, result = result_queue.get(timeout=timeout)
    except queue.Empty:
        return None

    if status == "ok":
        return result
    return None


def fetch_ohlcv_with_retry(ticker, count=CANDLE_COUNT, to=None, retries=3, delay=1):
    for _ in range(retries):
        timeout = 35 if count > 1_000 else 10
        df = get_ohlcv_with_timeout(ticker, count=count, to=to, timeout=timeout)
        if df is not None and not df.empty:
            return df
        time.sleep(delay)
    return None


def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def add_indicators(df):
    df = df.copy()
    df["ma20"] = df["close"].rolling(window=20).mean()
    df["ma60"] = df["close"].rolling(window=60).mean()
    df["ma200"] = df["close"].rolling(window=200).mean()
    df["rsi14"] = calculate_rsi(df["close"], 14)
    df["volume_avg20"] = df["volume"].rolling(window=20).mean()
    df["prev10_high"] = df["high"].shift(1).rolling(window=10).max()
    df["prev20_high"] = df["high"].shift(1).rolling(window=20).max()
    df["prev30_high"] = df["high"].shift(1).rolling(window=30).max()
    return df


def init_trades_file():
    if os.path.exists(TRADES_FILE):
        return

    columns = [
        "time",
        "ticker",
        "side",
        "price",
        "quantity",
        "cash_after",
        "profit_rate_percent",
        "fee_krw",
        "reason",
    ]
    pd.DataFrame(columns=columns).to_csv(TRADES_FILE, index=False, encoding="utf-8-sig")


def save_trade(ticker, side, price, quantity, cash_after, profit_rate, fee, reason):
    row = {
        "time": now_text(),
        "ticker": ticker,
        "side": side,
        "price": round(price, 4),
        "quantity": round(quantity, 10),
        "cash_after": round(cash_after, 2),
        "profit_rate_percent": "" if profit_rate is None else round(profit_rate * 100, 4),
        "fee_krw": round(fee, 4),
        "reason": reason,
    }
    pd.DataFrame([row]).to_csv(
        TRADES_FILE,
        mode="a",
        header=False,
        index=False,
        encoding="utf-8-sig",
    )


def required_indicators_ready(latest):
    required = ["ma20", "ma60", "ma200", "rsi14", "volume_avg20", "prev20_high"]
    return not pd.isna(latest[required]).any()


def is_buy_signal(latest):
    return (
        latest["close"] > latest["ma20"]
        and 45 <= latest["rsi14"] <= 75
        and latest["close"] > latest["open"]
        and latest["volume"] > latest["volume_avg20"] * 1.5
        and latest["close"] > latest["prev20_high"]
    )


def update_position_high(position, current_price):
    position["highest_price"] = max(position["highest_price"], current_price)


def calculate_position_profit_rate(position, current_price):
    sell_value_after_fee = current_price * position["quantity"] * (1 - FEE_RATE)
    return sell_value_after_fee / position["entry_cost"] - 1


def get_sell_reason(latest, position):
    current_price = latest["close"]
    update_position_high(position, current_price)

    profit_rate = calculate_position_profit_rate(position, current_price)
    trailing_rate = (current_price - position["highest_price"]) / position["highest_price"]

    if profit_rate >= TAKE_PROFIT:
        return "take_profit_1_2_percent"
    if profit_rate <= STOP_LOSS:
        return "stop_loss_minus_0_8_percent"
    if trailing_rate <= TRAILING_STOP:
        return "trailing_stop_minus_0_6_percent"
    if latest["rsi14"] >= 80:
        return "rsi_80_overheat_exit"
    if current_price < latest["ma20"]:
        return "price_below_ma20"

    return None


def get_latest_prices_from_market_rows(market_rows):
    prices = {}
    for ticker, row in market_rows.items():
        latest = row.get("latest")
        if latest is not None:
            prices[ticker] = latest["close"]
    return prices


def calculate_total_asset(cash, positions, latest_prices):
    position_value = 0
    for ticker, position in positions.items():
        price = latest_prices.get(ticker, position["buy_price"])
        position_value += price * position["quantity"] * (1 - FEE_RATE)
    return cash + position_value


def is_daily_loss_limit_reached(total_asset):
    return (total_asset - START_CASH) / START_CASH <= DAILY_MAX_LOSS


def can_reenter(ticker, latest_time, cooldown_until):
    blocked_until = cooldown_until.get(ticker)
    if blocked_until is None:
        return True
    return latest_time > blocked_until


def buy(ticker, latest, cash, positions, total_asset):
    trade_amount = min(total_asset * TRADE_RATIO, cash)
    if trade_amount <= 5_000:
        return cash

    price = latest["close"]
    fee = trade_amount * FEE_RATE
    quantity = (trade_amount - fee) / price
    cash -= trade_amount

    positions[ticker] = {
        "buy_price": price,
        "quantity": quantity,
        "buy_time": now_text(),
        "highest_price": price,
        "entry_cost": trade_amount,
    }

    save_trade(ticker, "BUY", price, quantity, cash, None, fee, "buy_signal")
    return cash


def sell(ticker, latest, cash, positions, reason):
    position = positions[ticker]
    price = latest["close"]
    quantity = position["quantity"]
    gross_value = price * quantity
    fee = gross_value * FEE_RATE
    net_value = gross_value - fee
    profit_rate = net_value / position["entry_cost"] - 1

    cash += net_value
    del positions[ticker]

    save_trade(ticker, "SELL", price, quantity, cash, profit_rate, fee, reason)
    return cash, profit_rate


def calculate_portfolio(cash, positions, market_rows):
    latest_prices = get_latest_prices_from_market_rows(market_rows)
    total_value = calculate_total_asset(cash, positions, latest_prices)
    position_value = total_value - cash
    total_profit = total_value - START_CASH
    total_profit_rate = total_profit / START_CASH * 100

    return {
        "cash": cash,
        "position_value": position_value,
        "total_value": total_value,
        "total_profit": total_profit,
        "total_profit_rate": total_profit_rate,
        "daily_loss_limit": is_daily_loss_limit_reached(total_value),
    }


def make_summary_panel(portfolio, last_update):
    profit_style = "green" if portfolio["total_profit"] >= 0 else "red"
    loss_status = "BUY BLOCKED" if portfolio["daily_loss_limit"] else "ACTIVE"
    loss_style = "red" if portfolio["daily_loss_limit"] else "green"

    table = Table.grid(expand=True)
    for _ in range(6):
        table.add_column(justify="center")

    table.add_row(
        "[bold]Total Asset[/bold]",
        "[bold]Cash[/bold]",
        "[bold]Position Value[/bold]",
        "[bold]Total PnL[/bold]",
        "[bold]Return[/bold]",
        "[bold]Daily Risk[/bold]",
    )
    table.add_row(
        f"[cyan]{format_krw(portfolio['total_value'])} KRW[/cyan]",
        f"{format_krw(portfolio['cash'])} KRW",
        f"{format_krw(portfolio['position_value'])} KRW",
        f"[{profit_style}]{format_krw(portfolio['total_profit'])} KRW[/{profit_style}]",
        f"[{profit_style}]{format_percent(portfolio['total_profit_rate'])}[/{profit_style}]",
        f"[{loss_style}]{loss_status}[/{loss_style}]",
    )

    title = f"{STRATEGY_NAME} | Fee included | {last_update}"
    return Panel(table, title=title, border_style="cyan", box=box.ROUNDED)


def make_coin_table(market_rows, positions, last_signals):
    table = Table(box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Ticker", style="bold")
    table.add_column("Price", justify="right")
    table.add_column("RSI", justify="right")
    table.add_column("MA20", justify="right")
    table.add_column("Prev20 High", justify="right")
    table.add_column("Position", justify="center")
    table.add_column("Entry", justify="right")
    table.add_column("Qty", justify="right")
    table.add_column("Unrealized PnL", justify="right")
    table.add_column("Return", justify="right")
    table.add_column("Last Signal", justify="center")

    for ticker in TICKERS:
        row = market_rows.get(ticker, {})
        latest = row.get("latest")
        error = row.get("error")
        position = positions.get(ticker)
        signal = last_signals.get(ticker, "WAIT")

        if signal.startswith("BUY"):
            signal_text = Text(signal, style="green bold")
        elif signal.startswith("SELL"):
            signal_text = Text(signal, style="red bold")
        elif signal.startswith("NO") or signal.startswith("BLOCK"):
            signal_text = Text(signal, style="yellow")
        else:
            signal_text = Text(signal, style="dim")

        if latest is None:
            table.add_row(
                ticker,
                "-",
                "-",
                "-",
                "-",
                Text("NO", style="dim"),
                "-",
                "-",
                "-",
                "-",
                Text(error or signal, style="yellow"),
            )
            continue

        price = latest["close"]

        if position:
            entry = position["buy_price"]
            quantity = position["quantity"]
            net_value = price * quantity * (1 - FEE_RATE)
            pnl = net_value - position["entry_cost"]
            profit_rate = pnl / position["entry_cost"] * 100
            pnl_style = "green" if pnl >= 0 else "red"
            position_text = Text("HOLD", style="green bold")
            entry_text = format_krw(entry)
            quantity_text = f"{quantity:.8f}"
            pnl_text = f"[{pnl_style}]{format_krw(pnl)}[/{pnl_style}]"
            profit_text = f"[{pnl_style}]{format_percent(profit_rate)}[/{pnl_style}]"
        else:
            position_text = Text("NO", style="dim")
            entry_text = "-"
            quantity_text = "-"
            pnl_text = "-"
            profit_text = "-"

        table.add_row(
            ticker,
            format_krw(price),
            f"{latest['rsi14']:.2f}",
            format_krw(latest["ma20"]),
            format_krw(latest["prev20_high"]),
            position_text,
            entry_text,
            quantity_text,
            pnl_text,
            profit_text,
            signal_text,
        )

    return Panel(table, title="Markets", border_style="white", box=box.ROUNDED)


def make_dashboard(cash, positions, market_rows, last_signals, last_update):
    portfolio = calculate_portfolio(cash, positions, market_rows)
    summary = make_summary_panel(portfolio, last_update)
    coin_table = make_coin_table(market_rows, positions, last_signals)
    footer = Panel(
        "Paper trading only. Real orders are NOT sent. Fee 0.05% per buy/sell included. Press Ctrl+C to stop.",
        border_style="dim",
        box=box.ROUNDED,
    )
    return Group(summary, coin_table, footer)


def print_plain_dashboard(cash, positions, market_rows, last_signals):
    portfolio = calculate_portfolio(cash, positions, market_rows)
    os.system("cls" if os.name == "nt" else "clear")
    print(f"{STRATEGY_NAME} | Fee included | {now_text()}")
    print(
        f"Total={format_krw(portfolio['total_value'])} Cash={format_krw(cash)} "
        f"Position={format_krw(portfolio['position_value'])} Return={format_percent(portfolio['total_profit_rate'])}"
    )
    print("-" * 120)
    print("Ticker     Price        RSI    MA20        Prev20High   Position   Entry       Qty          Signal")
    for ticker in TICKERS:
        latest = market_rows.get(ticker, {}).get("latest")
        position = positions.get(ticker)
        signal = last_signals.get(ticker, "WAIT")
        if latest is None:
            print(f"{ticker:<10} {'-':>10} {'-':>6} {'-':>10} {'-':>12} {'NO':>10} {'-':>10} {'-':>12} {signal}")
            continue
        print(
            f"{ticker:<10} {format_krw(latest['close']):>10} {latest['rsi14']:>6.2f} "
            f"{format_krw(latest['ma20']):>10} {format_krw(latest['prev20_high']):>12} "
            f"{('HOLD' if position else 'NO'):>10} "
            f"{(format_krw(position['buy_price']) if position else '-'):>10} "
            f"{(f'{position['quantity']:.8f}' if position else '-'):>12} {signal}"
        )


def get_yesterday_range():
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start = today - timedelta(days=1)
    end = today
    return start, end


def build_backtest_data(start, end):
    data = {}
    for ticker in TICKERS:
        print_line(f"Fetching {ticker}...")
        df = fetch_ohlcv_with_retry(
            ticker,
            count=BACKTEST_CANDLE_COUNT,
            to=end.strftime("%Y-%m-%d %H:%M:%S"),
            retries=3,
            delay=1,
        )
        if df is None:
            data[ticker] = None
            continue

        df = add_indicators(df)
        data[ticker] = df[(df.index >= start) & (df.index < end)]
    return data


def get_backtest_timestamps(data):
    timestamps = set()
    for df in data.values():
        if df is not None:
            timestamps.update(df.index)
    return sorted(timestamps)


def get_latest_backtest_prices(data, timestamp, fallback_prices):
    prices = dict(fallback_prices)
    for ticker, df in data.items():
        if df is not None and timestamp in df.index:
            prices[ticker] = df.loc[timestamp]["close"]
    return prices


def calculate_max_drawdown(equity_curve):
    if not equity_curve:
        return 0
    equity_series = pd.Series(equity_curve)
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak * 100
    return drawdown.min()


def calculate_max_consecutive_losses(closed_trades):
    max_losses = 0
    current_losses = 0
    for trade in closed_trades:
        if trade["profit_rate"] < 0:
            current_losses += 1
            max_losses = max(max_losses, current_losses)
        else:
            current_losses = 0
    return max_losses


def backtest():
    start, end = get_yesterday_range()
    print_line(f"[cyan]Running aggressive backtest:[/cyan] {start} ~ {end}" if RICH_AVAILABLE else f"Running aggressive backtest: {start} ~ {end}")

    data = build_backtest_data(start, end)
    timestamps = get_backtest_timestamps(data)

    cash = START_CASH
    positions = {}
    cooldown_until = {}
    latest_prices = {}
    equity_curve = []
    trades = []
    coin_stats = {
        ticker: {"trades": 0, "wins": 0, "realized_pnl": 0.0, "start_alloc": 0.0}
        for ticker in TICKERS
    }

    for timestamp in timestamps:
        latest_prices = get_latest_backtest_prices(data, timestamp, latest_prices)

        for ticker in list(positions.keys()):
            df = data.get(ticker)
            if df is None or timestamp not in df.index:
                continue
            latest = df.loc[timestamp]
            if not required_indicators_ready(latest):
                continue

            reason = get_sell_reason(latest, positions[ticker])
            if reason:
                position = positions[ticker]
                price = latest["close"]
                gross_value = price * position["quantity"]
                fee = gross_value * FEE_RATE
                net_value = gross_value - fee
                profit_rate = net_value / position["entry_cost"] - 1
                cash += net_value
                realized_pnl = net_value - position["entry_cost"]

                trades.append(
                    {
                        "ticker": ticker,
                        "time": timestamp,
                        "side": "SELL",
                        "price": price,
                        "quantity": position["quantity"],
                        "profit_rate": profit_rate,
                        "fee": fee,
                        "reason": reason,
                    }
                )
                coin_stats[ticker]["trades"] += 1
                coin_stats[ticker]["wins"] += 1 if profit_rate > 0 else 0
                coin_stats[ticker]["realized_pnl"] += realized_pnl
                del positions[ticker]
                cooldown_until[ticker] = timestamp + timedelta(minutes=5 * REENTRY_COOLDOWN_CANDLES)

        total_asset = calculate_total_asset(cash, positions, latest_prices)
        equity_curve.append(total_asset)
        buy_blocked = is_daily_loss_limit_reached(total_asset)

        if not buy_blocked and len(positions) < MAX_POSITIONS:
            for ticker in TICKERS:
                if len(positions) >= MAX_POSITIONS:
                    break
                if ticker in positions or not can_reenter(ticker, timestamp, cooldown_until):
                    continue

                df = data.get(ticker)
                if df is None or timestamp not in df.index:
                    continue
                latest = df.loc[timestamp]
                if not required_indicators_ready(latest):
                    continue
                if not is_buy_signal(latest):
                    continue

                total_asset = calculate_total_asset(cash, positions, latest_prices)
                trade_amount = min(total_asset * TRADE_RATIO, cash)
                if trade_amount <= 5_000:
                    continue

                price = latest["close"]
                fee = trade_amount * FEE_RATE
                quantity = (trade_amount - fee) / price
                cash -= trade_amount
                positions[ticker] = {
                    "buy_price": price,
                    "quantity": quantity,
                    "buy_time": timestamp,
                    "highest_price": price,
                    "entry_cost": trade_amount,
                }
                coin_stats[ticker]["start_alloc"] += trade_amount
                trades.append(
                    {
                        "ticker": ticker,
                        "time": timestamp,
                        "side": "BUY",
                        "price": price,
                        "quantity": quantity,
                        "profit_rate": None,
                        "fee": fee,
                        "reason": "buy_signal",
                    }
                )

    final_asset = calculate_total_asset(cash, positions, latest_prices)
    closed_trades = [trade for trade in trades if trade["side"] == "SELL"]
    total_trades = len(closed_trades)
    wins = sum(1 for trade in closed_trades if trade["profit_rate"] > 0)
    win_rate = wins / total_trades * 100 if total_trades else 0
    total_return = (final_asset - START_CASH) / START_CASH * 100
    avg_trade_return = (
        sum(trade["profit_rate"] for trade in closed_trades) / total_trades * 100
        if total_trades
        else 0
    )
    max_drawdown = calculate_max_drawdown(equity_curve)
    max_consecutive_losses = calculate_max_consecutive_losses(closed_trades)

    result = {
        "start": start,
        "end": end,
        "start_cash": START_CASH,
        "final_asset": final_asset,
        "total_return": total_return,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_return,
        "max_drawdown": max_drawdown,
        "max_consecutive_losses": max_consecutive_losses,
        "has_open_positions": len(positions) > 0,
        "open_positions": positions,
        "coin_stats": coin_stats,
        "data_errors": [ticker for ticker, df in data.items() if df is None],
    }
    print_backtest_result(result)


def print_backtest_result(result):
    if not RICH_AVAILABLE:
        print_plain_backtest_result(result)
        return

    profit_style = "green" if result["total_return"] >= 0 else "red"
    dd_style = "red" if result["max_drawdown"] < 0 else "green"

    summary = Table.grid(expand=True)
    for _ in range(4):
        summary.add_column(justify="center")

    summary.add_row("[bold]Start Cash[/bold]", "[bold]Final Asset[/bold]", "[bold]Total Return[/bold]", "[bold]Fee[/bold]")
    summary.add_row(
        f"{format_krw(result['start_cash'])} KRW",
        f"{format_krw(result['final_asset'])} KRW",
        f"[{profit_style}]{format_percent(result['total_return'])}[/{profit_style}]",
        "[cyan]0.05% buy/sell included[/cyan]",
    )
    summary.add_row("[bold]Trades[/bold]", "[bold]Win Rate[/bold]", "[bold]Avg Trade Return[/bold]", "[bold]Max Drawdown[/bold]")
    summary.add_row(
        str(result["total_trades"]),
        format_percent(result["win_rate"]),
        format_percent(result["avg_trade_return"]),
        f"[{dd_style}]{format_percent(result['max_drawdown'])}[/{dd_style}]",
    )
    summary.add_row("[bold]Max Consecutive Losses[/bold]", "[bold]Open Position[/bold]", "[bold]Strategy[/bold]", "[bold]Period[/bold]")
    summary.add_row(
        str(result["max_consecutive_losses"]),
        "YES" if result["has_open_positions"] else "NO",
        STRATEGY_NAME,
        f"{result['start']:%Y-%m-%d} ~ {result['end']:%Y-%m-%d}",
    )

    table = Table(title="Coin Performance", box=box.SIMPLE_HEAVY)
    table.add_column("Ticker", style="bold")
    table.add_column("Trades", justify="right")
    table.add_column("Wins", justify="right")
    table.add_column("Coin Return", justify="right")
    table.add_column("Realized PnL", justify="right")

    for ticker, stat in result["coin_stats"].items():
        coin_return = stat["realized_pnl"] / stat["start_alloc"] * 100 if stat["start_alloc"] else 0
        style = "green" if coin_return >= 0 else "red"
        table.add_row(
            ticker,
            str(stat["trades"]),
            str(stat["wins"]),
            f"[{style}]{format_percent(coin_return)}[/{style}]",
            f"[{style}]{format_krw(stat['realized_pnl'])} KRW[/{style}]",
        )

    title = f"Yesterday Backtest | {result['start']:%Y-%m-%d %H:%M} ~ {result['end']:%Y-%m-%d %H:%M}"
    console.clear()
    console.print(Panel(summary, title=title, border_style="cyan", box=box.ROUNDED))
    console.print(table)
    if result["data_errors"]:
        console.print(f"[yellow]Data fetch failed:[/yellow] {', '.join(result['data_errors'])}")


def print_plain_backtest_result(result):
    print("\nYesterday Backtest")
    print(f"Strategy: {STRATEGY_NAME}")
    print("Fee: 0.05% buy/sell included")
    print(f"Start Cash: {format_krw(result['start_cash'])} KRW")
    print(f"Final Asset: {format_krw(result['final_asset'])} KRW")
    print(f"Total Return: {format_percent(result['total_return'])}")
    print(f"Total Trades: {result['total_trades']}")
    print(f"Win Rate: {format_percent(result['win_rate'])}")
    print(f"Avg Trade Return: {format_percent(result['avg_trade_return'])}")
    print(f"Max Drawdown: {format_percent(result['max_drawdown'])}")
    print(f"Max Consecutive Losses: {result['max_consecutive_losses']}")
    print(f"Open Position: {'YES' if result['has_open_positions'] else 'NO'}")
    print("\nTicker     Trades  Wins  Coin Return  Realized PnL")
    for ticker, stat in result["coin_stats"].items():
        coin_return = stat["realized_pnl"] / stat["start_alloc"] * 100 if stat["start_alloc"] else 0
        print(
            f"{ticker:<10} {stat['trades']:>6} {stat['wins']:>5} "
            f"{format_percent(coin_return):>12} {format_krw(stat['realized_pnl']):>14} KRW"
        )


def get_recent_range(days):
    end = datetime.now().replace(second=0, microsecond=0)
    start = end - timedelta(days=days)
    return start, end


def build_optimizer_data(start, end):
    data = {}
    for ticker in TICKERS:
        print_line(f"Fetching 30d data {ticker}...")
        df = fetch_ohlcv_with_retry(
            ticker,
            count=OPTIMIZER_CANDLE_COUNT,
            to=end.strftime("%Y-%m-%d %H:%M:%S"),
            retries=3,
            delay=1,
        )
        if df is None:
            data[ticker] = None
            continue

        df = add_indicators(df)
        data[ticker] = df[(df.index >= start) & (df.index <= end)]
    return data


def make_strategy_candidates():
    candidates = []
    base_take_profits = [0.006, 0.009, 0.012, 0.015]
    base_stop_losses = [-0.004, -0.006, -0.008]
    volume_multipliers = [1.2, 1.5, 2.0]

    for take_profit in base_take_profits:
        for stop_loss in base_stop_losses:
            for volume_mult in volume_multipliers:
                candidates.append(
                    {
                        "name": "volume_breakout",
                        "take_profit": take_profit,
                        "stop_loss": stop_loss,
                        "trailing_stop": -0.006,
                        "volume_mult": volume_mult,
                        "rsi_low": 45,
                        "rsi_high": 75,
                        "high_window": 20,
                    }
                )
                candidates.append(
                    {
                        "name": "fast_breakout",
                        "take_profit": take_profit,
                        "stop_loss": stop_loss,
                        "trailing_stop": -0.005,
                        "volume_mult": volume_mult,
                        "rsi_low": 50,
                        "rsi_high": 80,
                        "high_window": 10,
                    }
                )

    for take_profit in [0.005, 0.008, 0.01]:
        for stop_loss in [-0.004, -0.006, -0.008]:
            candidates.append(
                {
                    "name": "rsi_rebound",
                    "take_profit": take_profit,
                    "stop_loss": stop_loss,
                    "trailing_stop": -0.005,
                    "volume_mult": 1.1,
                    "rsi_low": 35,
                    "rsi_high": 55,
                    "high_window": 20,
                }
            )
            candidates.append(
                {
                    "name": "ma20_reclaim",
                    "take_profit": take_profit,
                    "stop_loss": stop_loss,
                    "trailing_stop": -0.006,
                    "volume_mult": 1.0,
                    "rsi_low": 45,
                    "rsi_high": 70,
                    "high_window": 20,
                }
            )

    return candidates


def is_optimizer_buy_signal(latest, previous, params):
    if not required_indicators_ready(latest):
        return False

    name = params["name"]
    volume_ok = latest["volume"] > latest["volume_avg20"] * params["volume_mult"]
    candle_green = latest["close"] > latest["open"]
    rsi_ok = params["rsi_low"] <= latest["rsi14"] <= params["rsi_high"]
    high_col = f"prev{params['high_window']}_high"

    if name in ("volume_breakout", "fast_breakout"):
        return (
            latest["close"] > latest["ma20"]
            and rsi_ok
            and candle_green
            and volume_ok
            and latest["close"] > latest[high_col]
        )

    if name == "rsi_rebound":
        if previous is None or pd.isna(previous["rsi14"]):
            return False
        return (
            latest["close"] > latest["ma20"]
            and previous["rsi14"] < params["rsi_low"]
            and rsi_ok
            and candle_green
            and volume_ok
        )

    if name == "ma20_reclaim":
        if previous is None or pd.isna(previous["ma20"]):
            return False
        return (
            latest["close"] > latest["ma60"]
            and previous["close"] <= previous["ma20"]
            and latest["close"] > latest["ma20"]
            and rsi_ok
            and volume_ok
        )

    return False


def get_optimizer_sell_reason(latest, position, params):
    current_price = latest["close"]
    update_position_high(position, current_price)
    profit_rate = calculate_position_profit_rate(position, current_price)
    trailing_rate = (current_price - position["highest_price"]) / position["highest_price"]

    if profit_rate >= params["take_profit"]:
        return "take_profit"
    if profit_rate <= params["stop_loss"]:
        return "stop_loss"
    if trailing_rate <= params["trailing_stop"]:
        return "trailing_stop"
    if latest["rsi14"] >= 80:
        return "rsi_overheat"
    if current_price < latest["ma20"]:
        return "below_ma20"
    return None


def simulate_optimizer_candidate(data, timestamps, params):
    cash = START_CASH
    positions = {}
    cooldown_until = {}
    latest_prices = {}
    equity_curve = []
    closed_trades = []
    coin_stats = {ticker: {"trades": 0, "pnl": 0.0} for ticker in TICKERS}

    for timestamp in timestamps:
        latest_prices = get_latest_backtest_prices(data, timestamp, latest_prices)

        for ticker in list(positions.keys()):
            df = data.get(ticker)
            if df is None or timestamp not in df.index:
                continue

            latest = df.loc[timestamp]
            if not required_indicators_ready(latest):
                continue

            reason = get_optimizer_sell_reason(latest, positions[ticker], params)
            if reason:
                position = positions[ticker]
                price = latest["close"]
                gross_value = price * position["quantity"]
                fee = gross_value * FEE_RATE
                net_value = gross_value - fee
                profit_rate = net_value / position["entry_cost"] - 1
                cash += net_value
                pnl = net_value - position["entry_cost"]
                closed_trades.append({"ticker": ticker, "profit_rate": profit_rate, "pnl": pnl})
                coin_stats[ticker]["trades"] += 1
                coin_stats[ticker]["pnl"] += pnl
                del positions[ticker]
                cooldown_until[ticker] = timestamp + timedelta(minutes=5 * REENTRY_COOLDOWN_CANDLES)

        total_asset = calculate_total_asset(cash, positions, latest_prices)
        equity_curve.append(total_asset)
        if is_daily_loss_limit_reached(total_asset):
            continue

        for ticker in TICKERS:
            if len(positions) >= MAX_POSITIONS:
                break
            if ticker in positions or not can_reenter(ticker, timestamp, cooldown_until):
                continue

            df = data.get(ticker)
            if df is None or timestamp not in df.index:
                continue

            loc = df.index.get_loc(timestamp)
            previous = df.iloc[loc - 1] if loc > 0 else None
            latest = df.loc[timestamp]
            if not is_optimizer_buy_signal(latest, previous, params):
                continue

            total_asset = calculate_total_asset(cash, positions, latest_prices)
            trade_amount = min(total_asset * TRADE_RATIO, cash)
            if trade_amount <= 5_000:
                continue

            price = latest["close"]
            fee = trade_amount * FEE_RATE
            quantity = (trade_amount - fee) / price
            cash -= trade_amount
            positions[ticker] = {
                "buy_price": price,
                "quantity": quantity,
                "buy_time": timestamp,
                "highest_price": price,
                "entry_cost": trade_amount,
            }

    final_asset = calculate_total_asset(cash, positions, latest_prices)
    total_trades = len(closed_trades)
    wins = sum(1 for trade in closed_trades if trade["profit_rate"] > 0)
    return_rate = (final_asset - START_CASH) / START_CASH * 100
    win_rate = wins / total_trades * 100 if total_trades else 0
    avg_trade_return = (
        sum(trade["profit_rate"] for trade in closed_trades) / total_trades * 100
        if total_trades
        else 0
    )
    max_drawdown = calculate_max_drawdown(equity_curve)
    max_consecutive_losses = calculate_max_consecutive_losses(closed_trades)

    return {
        "params": params,
        "final_asset": final_asset,
        "return_rate": return_rate,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_return,
        "max_drawdown": max_drawdown,
        "max_consecutive_losses": max_consecutive_losses,
        "open_positions": len(positions),
        "coin_stats": coin_stats,
    }


def optimize_strategies():
    start, end = get_recent_range(OPTIMIZER_DAYS)
    print_line(
        f"[cyan]Running optimizer:[/cyan] {start} ~ {end}"
        if RICH_AVAILABLE
        else f"Running optimizer: {start} ~ {end}"
    )

    data = build_optimizer_data(start, end)
    timestamps = get_backtest_timestamps(data)
    candidates = make_strategy_candidates()

    results = []
    total = len(candidates)
    for index, params in enumerate(candidates, start=1):
        if index == 1 or index % 10 == 0 or index == total:
            print_line(f"Testing {index}/{total}...")
        results.append(simulate_optimizer_candidate(data, timestamps, params))

    results.sort(
        key=lambda item: (
            item["return_rate"],
            item["total_trades"],
            item["win_rate"],
            item["max_drawdown"],
        ),
        reverse=True,
    )
    print_optimizer_results(results, start, end, data)


def print_optimizer_results(results, start, end, data):
    data_errors = [ticker for ticker, df in data.items() if df is None]

    if not RICH_AVAILABLE:
        print_plain_optimizer_results(results, start, end, data_errors)
        return

    console.clear()
    table = Table(title="Top Strategy Optimizer Results | Fee 0.05% buy/sell included", box=box.SIMPLE_HEAVY)
    table.add_column("Rank", justify="right")
    table.add_column("Strategy", style="bold")
    table.add_column("Return", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Win", justify="right")
    table.add_column("Avg/Trade", justify="right")
    table.add_column("MDD", justify="right")
    table.add_column("TP/SL/Trail", justify="right")
    table.add_column("Vol", justify="right")
    table.add_column("RSI", justify="right")

    for rank, result in enumerate(results[:15], start=1):
        params = result["params"]
        style = "green" if result["return_rate"] >= 0 else "red"
        dd_style = "red" if result["max_drawdown"] < 0 else "green"
        table.add_row(
            str(rank),
            params["name"],
            f"[{style}]{format_percent(result['return_rate'])}[/{style}]",
            str(result["total_trades"]),
            format_percent(result["win_rate"]),
            format_percent(result["avg_trade_return"]),
            f"[{dd_style}]{format_percent(result['max_drawdown'])}[/{dd_style}]",
            f"{params['take_profit'] * 100:.1f}/{params['stop_loss'] * 100:.1f}/{params['trailing_stop'] * 100:.1f}",
            f"{params['volume_mult']:.1f}x",
            f"{params['rsi_low']}-{params['rsi_high']}",
        )

    best = results[0] if results else None
    title = f"30 Day Strategy Optimizer | {start:%Y-%m-%d} ~ {end:%Y-%m-%d}"
    console.print(Panel("Recent 30 days, 5min candles, paper trading, fee included", title=title, border_style="cyan"))
    console.print(table)

    if best:
        console.print(make_optimizer_coin_table(best))
    if data_errors:
        console.print(f"[yellow]Data fetch failed:[/yellow] {', '.join(data_errors)}")


def make_optimizer_coin_table(best):
    table = Table(title=f"Best Strategy Coin Stats: {best['params']['name']}", box=box.SIMPLE_HEAVY)
    table.add_column("Ticker", style="bold")
    table.add_column("Trades", justify="right")
    table.add_column("Realized PnL", justify="right")

    for ticker, stat in best["coin_stats"].items():
        style = "green" if stat["pnl"] >= 0 else "red"
        table.add_row(ticker, str(stat["trades"]), f"[{style}]{format_krw(stat['pnl'])} KRW[/{style}]")
    return table


def print_plain_optimizer_results(results, start, end, data_errors):
    print(f"\n30 Day Strategy Optimizer | {start:%Y-%m-%d} ~ {end:%Y-%m-%d}")
    print("Fee 0.05% buy/sell included")
    print("Rank Strategy          Return  Trades WinRate AvgTrade MDD     TP/SL/Trail Vol RSI")
    for rank, result in enumerate(results[:15], start=1):
        params = result["params"]
        print(
            f"{rank:>4} {params['name']:<16} {format_percent(result['return_rate']):>7} "
            f"{result['total_trades']:>6} {format_percent(result['win_rate']):>7} "
            f"{format_percent(result['avg_trade_return']):>8} {format_percent(result['max_drawdown']):>7} "
            f"{params['take_profit'] * 100:.1f}/{params['stop_loss'] * 100:.1f}/{params['trailing_stop'] * 100:.1f} "
            f"{params['volume_mult']:.1f}x {params['rsi_low']}-{params['rsi_high']}"
        )
    if data_errors:
        print(f"Data fetch failed: {', '.join(data_errors)}")


def run_bot():
    init_trades_file()
    cash = START_CASH
    positions = {}
    cooldown_until = {}
    market_rows = {}
    last_signals = {ticker: "WAIT" for ticker in TICKERS}

    if RICH_AVAILABLE:
        live_context = Live(
            make_dashboard(cash, positions, market_rows, last_signals, now_text()),
            refresh_per_second=4,
            screen=True,
        )
    else:
        live_context = None

    def update_screen():
        if RICH_AVAILABLE:
            live.update(make_dashboard(cash, positions, market_rows, last_signals, now_text()))
        else:
            print_plain_dashboard(cash, positions, market_rows, last_signals)

    context = live_context if RICH_AVAILABLE else DummyContext()
    with context:
        if RICH_AVAILABLE:
            live = live_context

        while True:
            try:
                for ticker in TICKERS:
                    try:
                        df = fetch_ohlcv_with_retry(ticker)
                        if df is None:
                            market_rows[ticker] = {"latest": None, "error": "DATA FAIL"}
                            last_signals[ticker] = "DATA FAIL"
                            continue

                        df = add_indicators(df)
                        latest = df.iloc[-1]
                        latest_time = df.index[-1]

                        if not required_indicators_ready(latest):
                            market_rows[ticker] = {"latest": None, "error": "INDICATOR WAIT"}
                            last_signals[ticker] = "INDICATOR WAIT"
                            continue

                        market_rows[ticker] = {"latest": latest, "error": None}
                        latest_prices = get_latest_prices_from_market_rows(market_rows)
                        total_asset = calculate_total_asset(cash, positions, latest_prices)
                        position = positions.get(ticker)
                        last_signals[ticker] = "WATCH"

                        if position:
                            reason = get_sell_reason(latest, position)
                            if reason:
                                cash, _ = sell(ticker, latest, cash, positions, reason)
                                cooldown_until[ticker] = latest_time + timedelta(
                                    minutes=5 * REENTRY_COOLDOWN_CANDLES
                                )
                                last_signals[ticker] = f"SELL {reason}"
                        else:
                            if is_buy_signal(latest):
                                if is_daily_loss_limit_reached(total_asset):
                                    last_signals[ticker] = "BLOCK DAILY LOSS"
                                elif len(positions) >= MAX_POSITIONS:
                                    last_signals[ticker] = "BLOCK MAX POSITIONS"
                                elif not can_reenter(ticker, latest_time, cooldown_until):
                                    last_signals[ticker] = "BLOCK COOLDOWN"
                                elif cash > 5_000:
                                    cash = buy(ticker, latest, cash, positions, total_asset)
                                    last_signals[ticker] = "BUY breakout"
                                else:
                                    last_signals[ticker] = "BUY SIGNAL CASH LOW"

                        update_screen()
                        time.sleep(0.2)

                    except Exception as e:
                        market_rows[ticker] = {
                            "latest": market_rows.get(ticker, {}).get("latest"),
                            "error": "ERROR",
                        }
                        last_signals[ticker] = f"ERROR {e}"

                update_screen()
                time.sleep(LOOP_SECONDS)

            except KeyboardInterrupt:
                break
            except Exception as e:
                last_signals["KRW-BTC"] = f"LOOP ERROR {e}"
                update_screen()
                time.sleep(LOOP_SECONDS)


class DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def choose_mode():
    if RICH_AVAILABLE:
        console.print(
            Panel(
                "[bold]1[/bold] Live paper trading\n"
                "[bold]2[/bold] Yesterday backtest\n"
                "[bold]3[/bold] 30 day strategy optimizer\n\n"
                "[cyan]Fee 0.05% buy/sell included. Real orders are never sent.[/cyan]"
            )
        )
    else:
        print("1 Live paper trading")
        print("2 Yesterday backtest")
        print("3 30 day strategy optimizer")
        print("Fee 0.05% buy/sell included. Real orders are never sent.")

    try:
        choice = input("Select mode (1 or 2): ").strip()
    except EOFError:
        choice = "1"

    if choice == "2":
        backtest()
    elif choice == "3":
        optimize_strategies()
    else:
        run_bot()


if __name__ == "__main__":
    choose_mode()
