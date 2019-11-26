from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# debug imports, remove later
import pandas as pd
import matplotlib.pyplot as plt

from Tagger import Tagger


class SentimentAnalyzer(object):
    def __init__(self, target_pos):
        self.tagger = Tagger(target_pos)
        self.classifier = LogisticRegression()

    def fit(self, X, y):
        X = self.tagger.create_bag_of_words(X)

        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, train_size=0.77)

        # self.classifier.fit(X_train, y_train)

        # prediction = self.classifier.predict(X_test)
        # plt.scatter(prediction, y_test, c='red')
        # plt.scatter(X_test, prediction, c='yellow')
        # plt.show()
