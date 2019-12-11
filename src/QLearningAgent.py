from enum import Enum
import numpy as np
import pandas as pd
import random

from TrendAnalyzer import TrendAnalyzer
from SentimentAnalyzer import SentimentAnalyzer


class Actions(Enum):
    BUY = 0
    SELL = 1
    HOLD = 2


class QLearningAgent(object):
    def __init__(self, starting_cash, number_of_stocks, epsilon=0.2, gamma=1, using_trend=False, using_sentiment=False, target_pos=None):
        self.cash_in_hand = starting_cash
        self.stocks_in_hand = np.zeros(number_of_stocks)
        self.opening_prices = np.zeros(number_of_stocks)
        self.closing_prices = np.zeros(number_of_stocks)
        self.initial_investment = starting_cash
        self.stock_trend = np.zeros(number_of_stocks)
        self.epsilon = epsilon
        self.gamma = gamma
        self.available_actions = [Actions.BUY, Actions.SELL, Actions.HOLD]

        self.curr_trend_analyzer = None
        self.curr_trend_score = None
        self.curr_sentiment_analyzer = None
        self.curr_sentiment_score = None

        # todo: add the analyzer constructors
        number_of_features = number_of_stocks + 1
        if using_trend:
            number_of_features += 1
            self.curr_trend_analyzer = TrendAnalyzer()
        if using_sentiment:
            number_of_features += 1
            self.curr_sentiment_analyzer = SentimentAnalyzer(target_pos)
        self.weights = np.zeros(number_of_features)

        # constants
        self.RANDOM_LIMIT = 100
        self.GLOBAL_OPEN_PRICE_MEAN = 0
        self.GLOBAL_OPEN_PRICE_RANGE = 0

    def update_stock_prices(self, opening_prices, closing_prices, curr_frame_row):
        self.opening_prices = opening_prices.iloc[curr_frame_row]
        if closing_prices is not None:
            self.closing_prices = closing_prices.iloc[curr_frame_row]

    def update_cash_and_stocks(self, cash, stocks):
        self.cash_in_hand = cash
        self.stocks_in_hand = stocks

    def explore_state_space(self):
        exploring_probability_limit = self.epsilon * self.RANDOM_LIMIT

        rand = np.random.randint(0, self.RANDOM_LIMIT)

        if rand <= exploring_probability_limit:
            return True
        return False

    def get_legal_actions(self):
        legal_actions = self.available_actions.copy()

        if self.cash_in_hand < self.opening_prices.min():
            legal_actions.remove(Actions.BUY)
        if not np.any(self.stocks_in_hand):
            legal_actions.remove(Actions.SELL)

        return legal_actions

    def choose_random_action(self):
        legal_actions = self.get_legal_actions()

        return random.choice(legal_actions)

    def choose_optimum_action(self):
        Q_s = dict()

        for action in self.get_legal_actions():
            cash, stocks = self.perform_action(action)
            feature = self.create_feature_vector(
                cash, self.opening_prices, stocks,)
            Q_s[action] = np.dot(self.weights, feature)

        return max(Q_s, key=Q_s.get)

    def create_feature_vector(self, cash, prices, stocks):
        cash /= self.initial_investment
        prices = (prices - self.GLOBAL_OPEN_PRICE_MEAN) / \
            self.GLOBAL_OPEN_PRICE_RANGE

        vec = [cash, np.mean(prices), np.mean(stocks)]
        if self.curr_sentiment_analyzer is not None:
            vec.append(self.curr_sentiment_score)
        if self.curr_trend_analyzer is not None:
            vec.append(self.curr_trend_score)

        return np.array(vec)

    def buy_stocks(self, cash, stocks):
        if self.stock_trend[0] > self.stock_trend[1]:
            buy_stock_1 = 0.7
            buy_stock_2 = 0.3
        elif self.stock_trend[0] < self.stock_trend[1]:
            buy_stock_1 = 0.3
            buy_stock_2 = 0.7
        else:
            buy_stock_1 = 0.5
            buy_stock_2 = 0.5

        n = 1
        while cash - (n * (buy_stock_1 * self.opening_prices[0] + buy_stock_2 * self.opening_prices[1])) > 0:
            n += 1
        n -= 1

        buy_stock_1 = int(buy_stock_1 * n)
        buy_stock_2 = n - buy_stock_1

        buy_stock = np.array([buy_stock_1, buy_stock_2])

        cash -= float(np.dot(buy_stock, self.opening_prices))
        stocks = stocks + buy_stock

        return cash, np.array(stocks)

    def calculate_reward(self):
        return float(np.dot(self.closing_prices, self.stocks_in_hand)) + self.cash_in_hand - self.initial_investment

    def generate_sample(self):
        Q_prime_sa = dict()

        for action in self.get_legal_actions():
            cash, stocks = self.perform_action(action)
            feature = self.create_feature_vector(
                cash, self.opening_prices, stocks)
            Q_prime_sa[action] = np.dot(self.weights, feature)

        return float(np.dot(self.closing_prices, self.stocks_in_hand)) + self.cash_in_hand - self.initial_investment + self.gamma * max(Q_prime_sa.values())

    def perform_action(self, action):
        cash = self.cash_in_hand
        stocks = self.stocks_in_hand.copy()

        if action is Actions.SELL:
            cash += np.dot(self.opening_prices, self.stocks_in_hand)
            stocks = np.zeros(self.stocks_in_hand.shape)
        elif action is Actions.BUY:
            cash, stocks = self.buy_stocks(cash, stocks)

        return cash, stocks

    def get_stock_trend(self, opening_prices, curr_row):
        df = opening_prices.iloc[list(range(curr_row, curr_row + 5))]
        diff = df.diff()
        stock_trend = diff.mean()

        return stock_trend

    def get_sentiment_score(self, news, date):
        date_idx = news.index
        date_idx = date_idx.where(date_idx <= date).tolist()
        date = [date_idx.index(date) for date in date_idx if type(
            date) is not pd._libs.tslibs.nattype.NaTType]
        date = date[-1]

        return self.curr_sentiment_analyzer.predict(news, date)

    def train_sentiment_analyzer(self, text, score):
        self.curr_sentiment_analyzer.fit(text, score)

    def train_agent(self, learning_rate=0.01, trials=10, opening_prices=None, closing_prices=None, news=None):
        if opening_prices is None and closing_prices is None:
            raise ReferenceError

        self.GLOBAL_OPEN_PRICE_MEAN = opening_prices.mean()
        self.GLOBAL_OPEN_PRICE_RANGE = opening_prices.max() - opening_prices.min()

        i = 0
        starting_cash = self.cash_in_hand
        while i < trials:
            curr_frame_row = 0
            for timestep in range(200):
                self.update_stock_prices(
                    opening_prices, closing_prices, curr_frame_row)

                self.stock_trend = self.get_stock_trend(
                    opening_prices, curr_frame_row)
                if news is not None and self.curr_sentiment_analyzer is not None:
                    date = opening_prices.index.values[curr_frame_row]
                    self.curr_sentiment_score = self.get_sentiment_score(
                        news, date)

                if self.explore_state_space():
                    action = self.choose_random_action()
                else:
                    action = self.choose_optimum_action()

                # update weights
                cash, stocks = self.perform_action(action)

                f_sa = self.create_feature_vector(
                    cash, self.opening_prices, stocks)
                Q_sa = float(np.dot(self.weights, f_sa))
                self.update_cash_and_stocks(cash, stocks)

                sample = self.generate_sample()
                difference = sample - Q_sa
                self.weights = self.weights + learning_rate * difference * f_sa

                curr_frame_row += 1

            self.reset_agent(starting_cash)

            i += 1
            print(self.weights)
        return self.weights

    def reset_agent(self, starting_cash):
        self.cash_in_hand = starting_cash
        self.initial_investment = starting_cash
        self.stocks_in_hand = np.array(
            [0 for i in range(self.stocks_in_hand.size)])
        self.opening_prices = np.array(
            [0 for i in range(self.opening_prices.size)])
        self.closing_prices = np.array(
            [0 for i in range(self.closing_prices.size)])

    def load_weights(self, weights):
        self.weights = np.array(weights)
