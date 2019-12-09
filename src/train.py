import pandas as pd
from sys import path
from QLearningAgent import QLearningAgent


gp_data = pd.read_csv(r'./data/GRAE.csv', infer_datetime_format=True,
                          index_col=0, parse_dates=True, memory_map=True)
square_data = pd.read_csv(r'./data/SQPH.csv', infer_datetime_format=True,
                          index_col=0, parse_dates=True, memory_map=True)

idx = gp_data.index.intersection(square_data.index)

opening_prices = pd.concat([gp_data.loc[idx, 'Open'].rename('GP').sort_index(), square_data.loc[idx, 'Open'].rename('SQPH').sort_index()], axis=1)
closing_prices = pd.concat([gp_data.loc[idx, 'Price'].rename('GP').sort_index(), square_data.loc[idx, 'Price'].rename('SQPH').sort_index()], axis=1)

print(opening_prices.head())
print(closing_prices.head())

agent = QLearningAgent(2, 3)
agent.set_initial_state(10000)

print(agent.train_agent(learning_rate=0.00000001, trials=100, opening_prices=opening_prices, closing_prices=closing_prices))
