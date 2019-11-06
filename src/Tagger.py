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
        processed_text = []

        for entry in raw_text:
            # remove everything except words
            entry = re.sub(r'[^a-zA-Z]', ' ', entry)
            # remove special characters
            entry = re.sub(r'\s+', ' ', entry)

            tokens = self.nlp(entry)

            # remove stopwords and single-character words from text
            tokens = [token.text.lower()
                      for token in tokens if not token.is_stop and len(token.text) > 1]
            tokens = " ".join(tokens)

            processed_text.append(tokens)

        return processed_text

    def pos_tag_text(self, processed_text):
        tagged_tokens = []

        for text in processed_text:
            tagged_text = self.nlp(text)

            l = []
            for token in tagged_text:
                if token.tag_ in self.target_pos:
                    l.append(token.text)

            tagged_tokens.append(l)

        return tagged_tokens

    def create_bag_of_words(self, text):
        processed_text = self.preprocess_text(text)
        tagged_tokens = self.pos_tag_text(processed_text)

        unique_tokens = list(set(chain.from_iterable(tagged_tokens)))
        bag_of_words = [[0 for i in range(len(unique_tokens))]
                        for j in range(len(tagged_tokens))]

        for row in range(len(tagged_tokens)):
            for entry in tagged_tokens[row]:
                if entry in unique_tokens:
                    bag_of_words[row][unique_tokens.index(entry)] = 1

        return pd.DataFrame(bag_of_words, columns=unique_tokens)
