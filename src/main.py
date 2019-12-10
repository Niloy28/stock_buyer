import pandas as pd

from SentimentAnalyzer import SentimentAnalyzer

target_pos = ['JJ', 'JJR', 'JJS', 'RB', 'RBR',
              'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'NNP', 'NNPS']

sentiment_analyzer = SentimentAnalyzer(target_pos)
dataframe = pd.read_csv(r'./data/sentiment.csv', memory_map=True, parse_dates=True,
                        infer_datetime_format=False, index_col=0, dtype={"News": str, "Score": float, "Binary Score": int})
df = pd.read_csv(r'./data/GRAE.csv')
print(df)

X = dataframe.loc[:, 'News']
y = dataframe.iloc[:, -1]

sentiment_analyzer.fit(X, y)
