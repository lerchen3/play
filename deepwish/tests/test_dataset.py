import torch

from train.train import ChatDataset


class DummyTokenizer:
    def __init__(self, vocab_size=32):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token = "</s>"
        self.pad_token = None

    def __len__(self):
        return self.vocab_size

    def apply_chat_template(self, messages, tokenize, return_tensors, padding, truncation, max_length):
        tokens = list(range(len(messages) * 2))
        tokens = (tokens + [self.pad_token_id] * max_length)[:max_length]
        return {"input_ids": torch.tensor(tokens).unsqueeze(0)}


class PlainTokenizer:
    def __init__(self, vocab_size=16):
        self.vocab_size = vocab_size
        self.pad_token_id = 1
        self.eos_token = "</s>"

    def __len__(self):
        return self.vocab_size

    def __call__(self, text, return_tensors, padding, truncation, max_length):
        tokens = list(range(min(len(text), max_length)))
        tokens = (tokens + [self.pad_token_id] * max_length)[:max_length]
        return {"input_ids": torch.tensor(tokens).unsqueeze(0)}


def test_chat_dataset_with_chat_template():
    tokenizer = DummyTokenizer()
    dataset = ChatDataset([
        "Hello",
    ], [
        "Hi",
    ], tokenizer, seq_len=8, mtp_depth=2)

    input_ids, tgt = dataset[0]
    assert input_ids.shape[0] == 8
    assert tgt.shape == (8, 2)
    # Ensure future tokens are copied correctly and padded at the end
    assert torch.all(tgt[-1] == tokenizer.pad_token_id)


def test_chat_dataset_without_chat_template():
    tokenizer = PlainTokenizer()
    dataset = ChatDataset(["Hi"], ["There"], tokenizer, seq_len=8, mtp_depth=0)

    input_ids = dataset[0]
    assert input_ids.shape[0] == 8
    # Plain tokenizer fallback should still pad somewhere in the sequence
    assert (input_ids == tokenizer.pad_token_id).any()
