import pandas as pd
from sys import path
from QLearningAgent import QLearningAgent


gp_data = pd.read_csv(r'./data/GRAE.csv', infer_datetime_format=True,
                          index_col=0, parse_dates=True, memory_map=True)
square_data = pd.read_csv(r'./data/SQPH.csv', infer_datetime_format=True,
                          index_col=0, parse_dates=True, memory_map=True)

idx = gp_data.index.intersection(square_data.index)

opening_prices = pd.concat([gp_data.loc[idx, 'Open'].rename('GP'), square_data.loc[idx, 'Open'].rename('SQPH')], axis=1)
closing_prices = pd.concat([gp_data.loc[idx, 'Price'].rename('GP'), square_data.loc[idx, 'Price'].rename('SQPH')], axis=1)

agent = QLearningAgent(2, 3)
agent.set_initial_state(10000)

print(agent.train_agent(trials=1000, opening_prices=opening_prices, closing_prices=closing_prices))
