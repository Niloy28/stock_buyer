import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from QLearningAgent import QLearningAgent

grae = pd.read_csv('./data/GRAE.csv', index_col=0, parse_dates=True)
sqph = pd.read_csv('./data/SQPH.csv', index_col=0, parse_dates=True)

idx = grae.index.intersection(sqph.index)
grae = grae.loc[idx].sort_index()
sqph = sqph.loc[idx].sort_index()

opening_prices = pd.concat([grae.loc[:, 'Open'].rename(
    'GP').sort_index(), sqph.loc[:, 'Open'].rename('SQPH').sort_index()], axis=1)
closing_prices = pd.concat([grae.loc[:, 'Price'].rename(
    'GP').sort_index(), sqph.loc[:, 'Price'].rename('SQPH').sort_index()], axis=1)

print(grae.head())
print(sqph.head())

for stock_df in (grae, sqph):
    stock_df['Norm Return'] = stock_df['Price'] / stock_df.iloc[0]['Price']

agent = QLearningAgent(2, 3)
agent.set_initial_state(10000)
# agent.load_weights([3492549400, 535662744, 232662777]) -> this gives ratio of 1.06
agent.load_weights([0.10864437, -0.09855167, 17.92765045])

grae_percent = []
sqph_percent = []

for curr_frame_row in range(opening_prices.shape[0]):
    agent.update_stock_prices(opening_prices, closing_prices, curr_frame_row)

    cash, stocks = agent.perform_action(agent.choose_optimum_action())
    grae_percent.append(stocks[0] / np.sum(stocks))
    sqph_percent.append(stocks[1] / np.sum(stocks))

grae_percent = np.mean(grae_percent)
sqph_percent = np.mean(sqph_percent)

# grae_percent = pd.DataFrame(grae_percent)
# sqph_percent = pd.DataFrame(sqph_percent)

for stock_df, allocation in zip((grae, sqph), (grae_percent, sqph_percent)):
    stock_df['Allocation'] = stock_df['Norm Return'] * allocation

for stock_df in (grae, sqph):
    stock_df['Position'] = stock_df['Allocation'] * 10000

all_pos = [grae['Position'], sqph['Position']]
# portf_val.columns = ['GRAE Pos', 'SQPH Pos']
portf_val = pd.concat(all_pos, axis=1)

portf_val['Total Pos'] = portf_val.sum(axis=1)
portf_val['Daily Return'] = portf_val['Total Pos'].pct_change(1)

sharpe_ratio = portf_val['Daily Return'].mean() / \
    portf_val['Daily Return'].std()
A_sharpe_ratio = sharpe_ratio * (252 ** 0.5)

print(A_sharpe_ratio)
