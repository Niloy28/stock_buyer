from enum import Enum
import pandas as pd 

from SentimentAnalyzer import SentimentAnalyzer


class Actions(Enum):
    BUY = 1
    SELL = 2
    HOLD = 3

class ReinforcementAgent(object):
    def __init__(self, stocks_in_hand, stock_closing_price, cash_in_hand):
        self.stocks_in_hand = stocks_in_hand
        self.cash_in_hand = cash_in_hand

    
    