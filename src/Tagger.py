import pandas as pd
import spacy
import re
from itertools import chain


class Tagger(object):
    def __init__(self, target_pos):
        # prepare language processor
        self.nlp = spacy.load("en_core_web_sm")
        self.target_pos = target_pos

    def preprocess_text(self, raw_text):
        # remove all non-alphabetic words and special characters
        processed_text = re.sub(r'[^a-zA-Z]', ' ', raw_text)
        processed_text = re.sub(r'\s+', ' ', processed_text)

        tokens = self.nlp(processed_text)
        tokens = [token.text.lower()
                  for token in tokens if not token.is_stop and len(token.text) > 1]

        processed_text = " ".join(tokens)

        return processed_text

    def pos_tag_text(self, processed_text):
        tagged_text = self.nlp(processed_text)
        tagged_tokens = []

        for entry in tagged_text:
            if entry.tag_ in self.target_pos:
                tagged_tokens.append(entry.text)

        return tagged_tokens

    def create_bag_of_words_for_fitting(self, text):
        processed_text = text.apply(self.preprocess_text)
        tagged_tokens = processed_text.apply(self.pos_tag_text)

        unique_tokens = list(set(chain.from_iterable(tagged_tokens)))
        bag_of_words = [[0 for i in range(len(unique_tokens))]
                        for j in range(len(tagged_tokens))]

        for row in range(len(tagged_tokens)):
            for entry in tagged_tokens[row]:
                if entry in unique_tokens:
                    bag_of_words[row][unique_tokens.index(entry)] = 1

        return pd.DataFrame(bag_of_words, columns=unique_tokens)

    def create_bag_of_words(self, text):
        processed_text = self.preprocess_text(text)
        tagged_tokens = self.pos_tag_text(processed_text)

        unique_tokens = list(set(tagged_tokens))
        bag_of_words = [[0 for i in range(len(unique_tokens))]
                        for j in range(len(tagged_tokens))]

        for row in range(len(tagged_tokens)):
            for entry in tagged_tokens[row]:
                if entry in unique_tokens:
                    bag_of_words[row][unique_tokens.index(entry)] = 1

        return pd.DataFrame(bag_of_words, columns=unique_tokens)
