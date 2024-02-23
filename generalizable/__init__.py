import os
import torch
from pydantic import BaseModel
from typing import Callable
import tiktoken
from custom_tokenizer import FrequencyGreedyTokenizer


def get_best_torch_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')

    return torch.device('cpu')


def get_batch(data, batch_size: int, block_size: int, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    # We include targets for each subsequence in each sequence in the batch
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


class TextDataset(BaseModel):
    all_tokens: list[int]
    vocab_size: int
    int_to_str: dict[int, str]
    decode_token_list_to_string: Callable[[[int]], str]


def load_text_directory(directory_path: str):
    contents = []
    real_path = os.path.expanduser(directory_path)
    print(f'Loading *.txt from: {real_path}')

    files_loaded = 0
    for file in os.listdir(real_path):
        if file.endswith(".txt"):
            # print(f'Loading: {file}')
            stripped_text = load_text_file(os.path.join(real_path, file))

            if "*** START OF THE PROJECT GUTENBERG EBOOK" in stripped_text:
                stripped_text = stripped_text.split("*** START OF THE PROJECT GUTENBERG EBOOK")[1]

            if "*** END OF THE PROJECT GUTENBERG EBOOK" in stripped_text:
                stripped_text = stripped_text.split("*** END OF THE PROJECT GUTENBERG EBOOK")[0]

            contents.append(stripped_text)
            files_loaded += 1

    raw_text = "\n\n\n".join(contents)

    print(f'Loaded {files_loaded} files, total length: {len(raw_text)}')

    return raw_text


def load_text_file(file_path):
    real_path = os.path.expanduser(file_path)
    with open(real_path, 'r', encoding='utf-8') as file:
        return file.read()


def encode_text(text, tokenizer):
    unique_tokens = []
    all_tokens = []
    int_to_str = {}

    if not isinstance(tokenizer, FrequencyGreedyTokenizer):
        raise ValueError(f'Unsupported tokenizer: {tokenizer}')

    all_raw_tokens = tokenizer.encode(text)
    unique_tokens = sorted(list(set(all_raw_tokens)))

    tok_to_int = {tok: i for i, tok in enumerate(unique_tokens)}
    int_to_str = {i: tokenizer.decode([tok]) for i, tok in enumerate(unique_tokens)}
    all_tokens = [tok_to_int[tok] for tok in all_raw_tokens]

    # if tokenization == 'char':
    #     unique_tokens = sorted(list(set(text)))
    #     tok_to_int = {tok: i for i, tok in enumerate(unique_tokens)}
    #     int_to_str = {i: tok for i, tok in enumerate(unique_tokens)}
    #     all_tokens = [tok_to_int[tok] for tok in text]
    # elif tokenization == 'tiktoken':
    #     enc = tiktoken.get_encoding("cl100k_base")
    #     all_raw_tokens = enc.encode(text)
    #     unique_tokens = sorted(list(set(all_raw_tokens)))
    #
    #     tok_to_int = {tok: i for i, tok in enumerate(unique_tokens)}
    #     int_to_str = {i: enc.decode([tok]) for i, tok in enumerate(unique_tokens)}
    #     all_tokens = [tok_to_int[tok] for tok in all_raw_tokens]
    # elif tokenization == 'bigram':
    #     if len(text) % 2 != 0:
    #         text = text + '\n'
    #     tokens = [text[i:i + 2] for i in range(0, len(text), 2)]
    #     unique_tokens = sorted(list(set(tokens)))
    #     tok_to_int = {tok: i for i, tok in enumerate(unique_tokens)}
    #     int_to_str = {i: tok for i, tok in enumerate(unique_tokens)}
    #     all_tokens = [tok_to_int[tok] for tok in tokens]
    # elif tokenization == 'trigram':
    #     while len(text) % 3 != 0:
    #         text = text + '\n'
    #     tokens = [text[i:i + 3] for i in range(0, len(text), 3)]
    #     unique_tokens = sorted(list(set(tokens)))
    #     tok_to_int = {tok: i for i, tok in enumerate(unique_tokens)}
    #     int_to_str = {i: tok for i, tok in enumerate(unique_tokens)}
    #     all_tokens = [tok_to_int[tok] for tok in tokens]
    # elif tokenization == 'trigram':
    #     pass

    vocab_size = len(unique_tokens)
    print(f'Text length: {len(text)}, total tokens: {len(all_tokens)}, vocab size: {vocab_size}')

    def decode_int_list_to_string(l: [int]) -> str:
        return ''.join([int_to_str[tok] for tok in l])

    return TextDataset(
        all_tokens=all_tokens,
        vocab_size=vocab_size,
        int_to_str=int_to_str,
        decode_token_list_to_string=decode_int_list_to_string,
    )


def estimate_loss(num_batches, get_batch, m):
    loss = 0.0
    for _ in range(num_batches):
        x, y = get_batch()
        logits, loss = m(x, y)
        loss += loss.item()

    return loss / num_batches

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
