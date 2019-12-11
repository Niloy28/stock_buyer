from QLearningAgent import QLearningAgent
import pandas as pd

# actual data for RL training
gp_data = pd.read_csv(r'./data/GRAE.csv', infer_datetime_format=True,
                      index_col=0, parse_dates=True, memory_map=True)
square_data = pd.read_csv(r'./data/SQPH.csv', infer_datetime_format=True,
                          index_col=0, parse_dates=True, memory_map=True)

# data for sentiment analysis
sentiment = pd.read_csv('./data/sentiment.csv', index_col=0,
                        parse_dates=True, memory_map=True)
target_pos = ['JJ', 'JJR', 'JJS', 'RB', 'RBR',
              'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'NNP', 'NNPS']
sentiment_text = sentiment.loc[:, 'News']
sentiment_score = sentiment.iloc[:, -1]

idx = gp_data.index.intersection(square_data.index)

opening_prices = pd.concat([gp_data.loc[idx, 'Open'].rename('GP').sort_index(
), square_data.loc[idx, 'Open'].rename('SQPH').sort_index()], axis=1)
closing_prices = pd.concat([gp_data.loc[idx, 'Price'].rename('GP').sort_index(
), square_data.loc[idx, 'Price'].rename('SQPH').sort_index()], axis=1)


agent = QLearningAgent(starting_cash=10000,
                       number_of_stocks=2, using_sentiment=False, target_pos=target_pos)

# agent.train_sentiment_analyzer(sentiment_text, sentiment_score)
agent.train_agent(learning_rate=0.00000001, trials=100, opening_prices=opening_prices,
                  closing_prices=closing_prices, news=sentiment_text)
