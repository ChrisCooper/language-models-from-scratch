from collections import Counter
import json


class FrequencyGreedyTokenizer:
    """
    I designed this before reding about tokenizers just to see what my own stab at it would look like.
    It counts frequencies of all 1,2,3,4,5,n-grams in the text and then selects the most frequent
    ones to be the vocabulary.
    """

    def __init__(self):
        self.max_token_length = 0
        self.vocab = []
        self.token_to_id = {}

    def save(self, vocab_path):
        with open(vocab_path, 'w') as file:
            json.dump(self.vocab, file)

    def load(self, vocab_path):
        with open(vocab_path, 'r') as file:
            self.vocab = json.load(file)

        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.max_token_length = max([len(token) for token in self.vocab])

    def train(self, training_text, vocab_size, min_frequency, max_token_length=5):
        self.max_token_length = max_token_length

        counters = {length: Counter() for length in self.token_lengths()}

        print(f"Counting combinations of characters in text of length {len(training_text)}...")
        char_pos = 0
        while char_pos < len(training_text) - 1:
            for token_length in self.token_lengths():
                if char_pos < len(training_text) - token_length:
                    counters[token_length][training_text[char_pos: char_pos + token_length]] += 1

            char_pos += 1

        # print(f"Found {len(singles)} unique singles, {len(bigrams)} unique bigrams, {len(trigrams)} unique trigrams")
        newline = '\n'
        print(f"{', '.join([f'{len(counter)} length {length} tokens' for length, counter in counters.items()])}")
        # for token_length in token_lengths():
        #     print(f"Top length {token_length} tokens: {newline.join([token for token, freq in counters[token_length].most_common(5)])}")

        # print(f"Top singles: {''.join([token for token, freq in singles.most_common(100)])}")
        # print(f"Top bigrams: {''.join([token for token, freq in bigrams.most_common(100)])}")
        # print(f"Top trigrams: {''.join([token for token, freq in trigrams.most_common(100)])}")

        # We have to be able to represent everything, so take all the singles to be sure
        self.vocab = list(counters[1].keys())
        remaining_vocab_size = vocab_size - len(self.vocab)

        print(f"Sorting tokens by frequency...")
        all_tokens_with_freqs = []
        for length, counter in counters.items():
            all_tokens_with_freqs += [(token, freq) for token, freq in counter.items()]

        all_tokens_with_freqs.sort(key=lambda x: x[1], reverse=True)

        # print(f"Top 30 tokens: {[token for token, freq in all_tokens_with_freqs[:30]]}")

        seen_vocab = set(self.vocab)
        for token, freq in all_tokens_with_freqs:
            if len(self.vocab) >= vocab_size:
                break
            if freq < min_frequency:
                break
            if token not in seen_vocab:
                self.vocab.append(token)
                seen_vocab.add(token)

        # Sort vocab by frequency
        def get_token_freq(token):
            counter = counters[len(token)]
            return counter[token]

        self.vocab.sort(key=lambda token: get_token_freq(token), reverse=True)

        # Create a dictionary of tokens to their ID (index in the vocabulary)
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}

        # num_singles, num_bigrams, num_trigrams = 0, 0, 0
        # for token in self.vocab:
        #     if len(token) == 1:
        #         num_singles += 1
        #     elif len(token) == 2:
        #         num_bigrams += 1
        #     else:
        #         num_trigrams += 1
        #
        # print(f"Vocab size: {len(self.vocab)}, singles: {num_singles}, bigrams: {num_bigrams}, trigrams: {num_trigrams}")
        vocab_token_size_count = Counter([len(token) for token in self.vocab])

        print(
            f"Vocab size: {len(self.vocab)}, {', '.join([f'{count} length {length} tokens' for length, count in vocab_token_size_count.items()])}")

    def token_lengths(self):
        for i in range(1, self.max_token_length + 1):
            yield i

    def encode(self, text):
        i = 0
        encoded = []
        while i < len(text) - 1:
            # Prioritize longer tokens
            for token_length in range(self.max_token_length, 0, -1):
                if i < len(text) - token_length and text[i: i + token_length] in self.token_to_id:
                    encoded.append(self.token_to_id[text[i: i + token_length]])
                    i += token_length
                    break

        return encoded

    def decode(self, encoded):
        return ''.join([self.vocab[i] for i in encoded])
