import numpy as np


def _simulate_trading_buy_and_hold(true_rets, fee_rate):
    balance = 1.0
    trade_stats = []
    balance -= balance * fee_rate # buy 
    for t in range(len(true_rets)):
        balance += balance * true_rets[t]
    balance -= balance * fee_rate # sell at the last day 
    profit_rate = balance - 1.0
    trade_stats = [[len(true_rets)-1, balance, profit_rate]]
    return balance - 1.0, trade_stats


def _simulate_trading_daily_buy_sell(true_rets, pred_rets, buy_threshold, fee_rate):
    balance = 1.0
    trade_stats = []
    for t in range(len(pred_rets)):
        if pred_rets[t] > buy_threshold: 
            orig_balance = balance
            balance -= balance * fee_rate # buy
            balance += balance * true_rets[t]
            balance -= balance * fee_rate # sell
            profit_rate = (balance - orig_balance) / orig_balance
            trade_stats.append([t, balance, profit_rate]) 
    return balance - 1.0, trade_stats


def _simulate_trading_buy_hold_sell(true_rets, pred_rets, buy_threshold, sell_threshold, fee_rate):
    balance = 1.0
    trade_stats = []
    holding = False
    profit_rate = 0.0
    for t in range(len(pred_rets)):
        if not holding:
            if pred_rets[t] > buy_threshold: # buy 
                orig_balance = balance
                balance -= balance * fee_rate
                balance += balance * true_rets[t]
                holding = True
        else: 
            if (pred_rets[t] < sell_threshold) or (t == len(pred_rets)-1): # sell
                balance += balance * true_rets[t]
                balance -= balance * fee_rate
                profit_rate = (balance - orig_balance) / orig_balance
                trade_stats.append([t, balance, profit_rate])
                holding = False
            else:
                balance += balance * true_rets[t]

    assert holding == False
    return balance - 1.0, trade_stats


def simulate_trading(true_rets, pred_rets, strategy='daily_buy_sell', buy_threshold=0.002, sell_threshold=0.0, fee_rate=0.001):
    assert strategy == 'buy_and_hold' or strategy == 'daily_buy_sell' or strategy == 'buy_hold_sell'
    assert len(true_rets) == len(pred_rets)

    if strategy == 'buy_and_hold':
        accumulated_ret, trade_stats = _simulate_trading_buy_and_hold(true_rets, fee_rate)
    elif strategy == 'daily_buy_sell':
        accumulated_ret, trade_stats = _simulate_trading_daily_buy_sell(true_rets, pred_rets, buy_threshold, fee_rate)
    else: # strategy == 'buy_hold_sell':
        accumulated_ret, trade_stats = _simulate_trading_buy_hold_sell(true_rets, pred_rets, buy_threshold, sell_threshold, fee_rate)
        
    num_of_trades = len(trade_stats)
    trade_stats = np.array(trade_stats)
    num_of_successful_trades = np.count_nonzero(trade_stats[:, 2] > 0) if num_of_trades > 0 else 0
    mean_profit_rate = np.mean(trade_stats[:, 2]) if num_of_trades > 0 else 0.0
    return accumulated_ret, mean_profit_rate, num_of_trades, num_of_successful_trades
