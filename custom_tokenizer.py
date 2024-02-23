from collections import Counter
import json
import time


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


class BytePairEncodingTokenizer:
    def __init__(self):
        self.substitution_order_for_encoding = []
        self.substitution_map = {}
        self.vocab_size = 0

    def train(self, training_text, max_vocab_size, min_frequency):
        sequence = list(training_text.encode('utf-8'))
        original_sequence_length = len(sequence)
        next_sub_value = max(sequence) + 1
        self.vocab_size = len(set(sequence))

        iteration = 0
        last_logged_time = time.time()
        while True:
            # print(f"Iteration {iteration}, sequence length: {len(sequence)}")
            counter = Counter()
            char_pos = 0
            while char_pos < len(sequence) - 2:
                counter[(sequence[char_pos], sequence[char_pos + 1])] += 1
                char_pos += 1

            most_common_pair, most_common_count = counter.most_common(1)[0]

            if most_common_count < min_frequency:
                # print(f"Stopping token substitution at {most_common_pair} ({self.str_repr(most_common_pair)}), frequency({most_common_count}) < min_frequency({min_frequency})")
                break

            # To allow more efficient decoding, we should expand out recursive substitutions in the map
            # This isn't supposed to be an efficient implementation, so we'll just do it the naive way
            self.substitution_map[next_sub_value] = most_common_pair
            self.substitution_order_for_encoding.append(next_sub_value)
            self.vocab_size += 1

            if time.time() - last_logged_time > 10:
                print(f"Iteration {iteration}: Substituted {most_common_pair} ({self.str_repr(most_common_pair)}) (seen {most_common_count} times) with {next_sub_value}")
                last_logged_time = time.time()

            sequence = self.substitute_pair(sequence, pair_to_sub=most_common_pair, replacement=next_sub_value)

            next_sub_value += 1
            iteration += 1

            if self.vocab_size >= max_vocab_size:
                # print(f"Vocab size reached: {self.vocab_size} >= {max_vocab_size}")
                break

        print(f"Vocab size: {self.vocab_size}, sequence length: {len(sequence)} (originally: {original_sequence_length})")
        # print(f"Substitution map: {self.substitution_map}")
        # print(f"Substitution order for encoding: {self.substitution_order_for_encoding}")
        # print(f"New sequence:\n{sequence}")
        # print(f"Decoded: {self.decode(sequence)}")

    def substitute_pair(self, sequence, pair_to_sub, replacement):
        new_sequence = []
        char_pos = 0
        while char_pos < len(sequence) - 1:
            if (sequence[char_pos], sequence[char_pos + 1]) == pair_to_sub:
                new_sequence.append(replacement)
                char_pos += 2
            else:
                new_sequence.append(sequence[char_pos])
                char_pos += 1

        if char_pos < len(sequence):
            # Add the last character if it wasn't part of a pair
            new_sequence.append(sequence[char_pos])

        return new_sequence

    def str_repr(self, sequence):
        try:
            return bytes(list(sequence)).decode('utf-8')
        except BaseException as e:
            return "not a valid UTF-8 sequence"


    def encode(self, text):
        sequence = list(text.encode('utf-8'))

        for sub_value in self.substitution_order_for_encoding:
            pair_to_sub = self.substitution_map[sub_value]
            sequence = self.substitute_pair(sequence, pair_to_sub=pair_to_sub, replacement=sub_value)

        return sequence

    def expand_token(self, sequence, token, replacement_pair):
        new_sequence = []
        char_pos = 0
        while char_pos < len(sequence):
            if sequence[char_pos] == token:
                new_sequence.append(replacement_pair[0])
                new_sequence.append(replacement_pair[1])
            else:
                new_sequence.append(sequence[char_pos])
            char_pos += 1

        return new_sequence

    def decode(self, token_list: list[int]):
        # print(f"Decoding:\n{token_list}")
        sequence = token_list
        for sub_value in reversed(self.substitution_order_for_encoding):
            pair_to_sub = self.substitution_map[sub_value]
            # print(f"Expanding {sub_value} -> {pair_to_sub} ({self.str_repr(pair_to_sub)})")
            sequence = self.expand_token(sequence, sub_value, pair_to_sub)

        return bytes(sequence).decode('utf-8', errors='replace')



