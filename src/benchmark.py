import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from QLearningAgent import QLearningAgent

grae = pd.read_csv('./data/GRAE.csv', index_col=0, parse_dates=True)
sqph = pd.read_csv('./data/SQPH.csv', index_col=0, parse_dates=True)

idx = grae.index.intersection(sqph.index)
grae = grae.loc[idx].sort_index().head(n=252)
sqph = sqph.loc[idx].sort_index().head(n=252)

opening_prices = pd.concat([grae.loc[:, 'Open'].rename(
    'GP').sort_index(), sqph.loc[:, 'Open'].rename('SQPH').sort_index()], axis=1)
closing_prices = pd.concat([grae.loc[:, 'Price'].rename(
    'GP').sort_index(), sqph.loc[:, 'Price'].rename('SQPH').sort_index()], axis=1)


for stock_df in (grae, sqph):
    stock_df['Norm Return'] = stock_df['Price'] / stock_df.iloc[0]['Price']

agent = QLearningAgent(starting_cash=10000, number_of_stocks=2)
agent.load_weights([0.15952694, -0.12913071, 26.15459896])

grae_percent = []
sqph_percent = []

for curr_frame_row in range(opening_prices.shape[0]):
    agent.update_stock_prices(opening_prices, closing_prices, curr_frame_row)

    cash, stocks = agent.perform_action(agent.choose_optimum_action())
    grae_percent.append(stocks[0] / np.sum(stocks))
    sqph_percent.append(stocks[1] / np.sum(stocks))

for stock_df, allocation in zip((grae, sqph), (grae_percent, sqph_percent)):
    stock_df['Allocation'] = stock_df['Norm Return'] * allocation

for stock_df in (grae, sqph):
    stock_df['Position'] = stock_df['Allocation'] * 10000

all_pos = [grae['Position'], sqph['Position']]
portf_val = pd.concat(all_pos, axis=1)

portf_val['Total Pos'] = portf_val.sum(axis=1)
portf_val['Daily Return'] = portf_val['Total Pos'].pct_change(1)

sharpe_ratio = portf_val['Daily Return'].mean() / \
    portf_val['Daily Return'].std()
A_sharpe_ratio = sharpe_ratio * (252 ** 0.5)

print(A_sharpe_ratio)
