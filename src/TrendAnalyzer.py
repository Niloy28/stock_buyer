from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense


class TrendAnalyzer(object):
    def __init__(self, parameter_list):
        # Initialising the ANN
        self.classifier = Sequential()

        # Adding the input layer and the first hidden layer
        self.classifier.add(Dense(output_dim=6, init='uniform',
                                  activation='relu', input_dim=5))

        # Adding the second hidden layer
        self.classifier.add(
            Dense(output_dim=6, init='uniform', activation='relu'))

        # Adding the output layer
        self.classifier.add(
            Dense(output_dim=1, init='uniform', activation='sigmoid'))

        # Compiling the ANN
        self.classifier.compile(
            optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.classifier.summary()

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0)

        sc = StandardScaler()

        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Fitting the ANN to the Training set
        a = classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)

        # Part 3 - Making the predictions and evaluating the model
        a.history
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        y_pred = (y_pred > 0.5)

        cm = confusion_matrix(y_test, y_pred)

    def predict(self, X):
        return self.classifier.predict(X)
