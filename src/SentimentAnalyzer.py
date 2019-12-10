from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from Tagger import Tagger
import numpy as np
import pandas as pd


class SentimentAnalyzer(object):
    def __init__(self, target_pos):
        self.tagger = Tagger(target_pos)
        self.classifier = LogisticRegression(random_state=0)
        self.bag_of_words = None

    def fit(self, X, y):
        self.bag_of_words = self.tagger.create_bag_of_words_for_fitting(X)
        print(self.bag_of_words)
        X = self.bag_of_words

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=0)

        # feature scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        self.classifier.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = self.classifier.predict(X_test)

        score = self.classifier.score(X_test, y_test)
        print("test set accuracy is: ", score*100, "%")

        # Making the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("confusion matrix is as follow-")
        print(cm)

    def predict(self, X, idx):
        table = self.tagger.create_bag_of_words(X.iloc[idx])
        entry = np.array(self.bag_of_words[np.array(table) == 1])

        return self.classifier.predict(pd.array(entry))
