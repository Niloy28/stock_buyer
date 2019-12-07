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
    def __init__(self, number_of_stocks, number_of_features, epsilon=0.2, gamma=1, using_trend=False, using_sentiment=False):
        self.cash_in_hand = 0
        self.stocks_in_hand = np.array([0 for i in range(number_of_stocks)])
        self.opening_prices = np.array([0 for i in range(number_of_stocks)])
        self.closing_prices = np.array([0 for i in range(number_of_stocks)])
        self.initial_investment = 0
        self.epsilon = epsilon
        self.gamma = gamma
        self.available_actions = [Actions.BUY, Actions.SELL, Actions.HOLD]

        # todo: add the analyzer constructors
        if using_trend:
            number_of_features += 1
        if using_sentiment:
            number_of_features += 1
        self.weights = np.zeros(number_of_features)

        # constants
        self.RANDOM_LIMIT = 100
        self.GLOBAL_OPEN_PRICE_MEAN = 0
        self.GLOBAL_OPEN_PRICE_RANGE = 0

    def set_initial_state(self, starting_cash):
        self.cash_in_hand = starting_cash
        self.initial_investment = starting_cash

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
            Q_s[action] = self.calculate_Q_sa(cash, stocks)

        return max(Q_s, key=Q_s.get)

    def create_feature_vector(self, cash, prices, stocks):
        cash /= self.initial_investment
        prices = (prices - self.GLOBAL_OPEN_PRICE_MEAN) / self.GLOBAL_OPEN_PRICE_RANGE

        return np.array([cash, np.mean(prices), np.mean(stocks)])

    def buy_stocks(self, cash, stocks):
        while cash >= min(self.opening_prices):
            idx = np.random.randint(0, len(stocks))
            if self.opening_prices[idx] > cash:
                continue
            stocks[idx] += 1
            cash -= self.opening_prices[idx]

        return cash, np.array(stocks)
        
    def calculate_reward(self):
        return float(np.dot(self.closing_prices, self.stocks_in_hand)) + self.cash_in_hand - self.initial_investment

    def generate_sample(self):
        Q_prime_sa = dict()

        for action in self.get_legal_actions():
            cash, stocks = self.perform_action(action)
            Q_prime_sa[action] = self.calculate_Q_sa(cash, stocks)

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

    def calculate_Q_sa(self, cash, stocks):
        features = self.create_feature_vector(cash, self.opening_prices, stocks)

        return float(np.dot(self.weights, features))

    def update_weights(self, action, learning_rate):
        cash, stocks = self.perform_action(action)

        f_sa = self.create_feature_vector(cash, self.opening_prices, stocks)
        Q_sa = self.calculate_Q_sa(cash, stocks)
        self.update_cash_and_stocks(cash, stocks)

        sample = self.generate_sample()
        difference = sample - Q_sa
        self.weights = self.weights + learning_rate * difference * f_sa

    def train_agent(self, learning_rate=0.01, trials=10, opening_prices=None, closing_prices=None):
        if opening_prices is None:
            raise ReferenceError

        self.GLOBAL_OPEN_PRICE_MEAN = opening_prices.mean()
        self.GLOBAL_OPEN_PRICE_RANGE = opening_prices.max() - opening_prices.min()

        i = 0
        starting_cash = self.cash_in_hand
        while i < trials:
            curr_frame_row = 0
            for timestep in range(100):
                self.update_stock_prices(opening_prices, closing_prices, curr_frame_row)

                if self.explore_state_space():
                    action = self.choose_random_action()
                else:
                    action = self.choose_optimum_action()
                    
                # update weights
                cash, stocks = self.perform_action(action)

                f_sa = self.create_feature_vector(cash, self.opening_prices, stocks)
                Q_sa = float(np.dot(self.weights, f_sa))
                self.update_cash_and_stocks(cash, stocks)

                sample = self.generate_sample()
                difference = sample - Q_sa
                self.weights = self.weights + learning_rate * difference * f_sa

                curr_frame_row += 1

            self.reset_agent(starting_cash)

            i += 1
        return self.weights

    def reset_agent(self, starting_cash):
        self.cash_in_hand = starting_cash
        self.initial_investment = starting_cash
        self.stocks_in_hand = np.array([0 for i in range(self.stocks_in_hand.size)])
        self.opening_prices = np.array([0 for i in range(self.opening_prices.size)])
        self.closing_prices = np.array([0 for i in range(self.closing_prices.size)])

    def load_weights(self, weights):
        self.weights = np.array(weights)