from enum import Enum
import numpy as np
import pandas as pd

from SentimentAnalyzer import SentimentAnalyzer


class Actions(Enum):
    BUY = 1
    SELL = 2
    HOLD = 3


class QLearningAgent(object):
    def __init__(self, number_of_stocks, number_of_features, epsilon=0.2, gamma=1, learning_rate=0.01, using_trend=False, using_sentiment=False):
        self.cash_in_hand = 0
        self.stocks_in_hand = [0 for i in range(number_of_stocks)]
        self.closing_prices = [0 for i in range(number_of_stocks)]
        self.investment = 0
        self.epsilon = epsilon

        # todo: add the analyzer constructors
        if using_trend:
            number_of_features += 1
        if using_sentiment:
            number_of_features += 1
        self.weights = np.array([0.0 for i in range(number_of_features)])

    def set_initial_state(self, starting_cash, initial_closing_prices):
        self.cash_in_hand = starting_cash
        self.closing_prices = initial_closing_prices

    def generate_sample(self):
        return self.calculate_reward() + self.gamma

    def calculate_reward(self):
        return np.multiply(self.closing_prices, self.stocks_in_hand) - self.investment

    def update_weights(self):
        features = np.array([self.cash_in_hand, sum(
            self.closing_prices), sum(self.stocks_in_hand)])

        Q_s = self.weights * features.T
