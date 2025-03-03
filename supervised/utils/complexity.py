from collections import Counter

import numpy as np
import pandas as pd
import spacy


# Load spaCy model for NLP tasks
nlp = spacy.load('en_core_web_sm')
# may need to run python -m spacy download en_core_web_sm

class ComplexityFeatures:
    def __init__(self):
        pass

    def calculate_entropy(self, text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 0
        words = text.lower().split()
        word_counts = Counter(words)
        total_words = sum(word_counts.values())
        probabilities = [count / total_words for count in word_counts.values()]
        return -sum(p * np.log2(p) for p in probabilities)

    def lexical_complexity(self, text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 0
        words = text.lower().split()
        return len(set(words)) / len(words) if words else 0

    def pos_complexity(self, text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 0
        doc = nlp(text)
        return sum(1 for token in doc if token.pos_ in ["ADJ", "NOUN"])

    def medical_entity_count(self, text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 0
        doc = nlp(text)
        return len([ent for ent in doc.ents])

    def text_features(self, text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return [0] * 6
        return [
            self.calculate_entropy(text),
            self.lexical_complexity(text),
            self.pos_complexity(text),
            self.medical_entity_count(text),
            len(text),
            len(text.split())
        ]