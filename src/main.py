import pandas as pd

from SentimentAnalyzer import SentimentAnalyzer

target_pos = ['JJ', 'JJR', 'JJS', 'RB', 'RBR',
              'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'NNP', 'NNPS']

sentiment_analyzer = SentimentAnalyzer(target_pos)
dataframe = pd.read_csv('./data/sentiment.csv')
X = dataframe.loc[:, 'News']
y = dataframe.iloc[:, -1]

sentiment_analyzer.fit(X, y)
