import numpy as np
import math
import random
import pickle
from collections import defaultdict


def _calculate_mean_probabilities(chain):
    out_mean = {word: np.mean(list(transitions.values())) if transitions else 0 for word, transitions in
                chain.items()}
    in_transitions = defaultdict(list)
    for word, transitions in chain.items():
        for next_word, prob in transitions.items():
            in_transitions[next_word].append(prob)
    in_mean = {word: np.mean(in_transitions[word]) if word in in_transitions else 0 for word in chain}
    return in_mean, out_mean


def _build_probabilistic_chain(documents):
    chain = defaultdict(lambda: defaultdict(int))
    for doc in documents:
        tokens = doc.lower().split()
        for i in range(len(tokens) - 1):
            chain[tokens[i]][tokens[i + 1]] += 1
    for word, transitions in chain.items():
        total = sum(transitions.values())
        for next_word in transitions:
            transitions[next_word] /= total
    return dict(chain)


class GraphProbabilisticInformationLeakageDetector:
    def __init__(self, p1=2, p2=2, smoothing_constant=1e-8):
        self.p1 = p1
        self.p2 = p2
        self.smoothing_constant = smoothing_constant
        self.normal_chain = None
        self.leak_chain = None
        self.normal_in_mean = None
        self.leak_in_mean = None
        self.normal_out_mean = None
        self.leak_out_mean = None

    def _calculate_sequence_probability(self, sequence, chain, in_mean, out_mean):
        log_prob = 0.0
        for i in range(len(sequence) - 1):
            current_word, next_word = sequence[i].lower(), sequence[i + 1].lower()
            if current_word in chain and next_word in chain[current_word]:
                transition_prob = chain[current_word][next_word]
            else:
                out_prob = out_mean.get(current_word, self.smoothing_constant) ** self.p1
                in_prob = in_mean.get(next_word, self.smoothing_constant) ** self.p2
                transition_prob = out_prob * in_prob
            log_prob += math.log10(transition_prob) if transition_prob > 0 else -15
        return log_prob

    def train(self, normal_text, leaked_text):
        self.normal_chain = _build_probabilistic_chain(normal_text)
        self.leak_chain = _build_probabilistic_chain(leaked_text)
        self.normal_in_mean, self.normal_out_mean = _calculate_mean_probabilities(self.normal_chain)
        self.leak_in_mean, self.leak_out_mean = _calculate_mean_probabilities(self.leak_chain)

    def predict(self, text):
        tokens = text.lower().split()
        if len(tokens) < 2:
            return random.randint(0, 1)
        real_prob = self._calculate_sequence_probability(tokens, self.normal_chain, self.normal_in_mean, self.normal_out_mean)
        fake_prob = self._calculate_sequence_probability(tokens, self.leak_chain, self.leak_in_mean, self.leak_out_mean)
        return 1 if fake_prob > real_prob else 0

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)