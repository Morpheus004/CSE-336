from os import replace
from typing import Iterable
import regex as re


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # mapping from bytes to int. Opposite of vocab
        self.id = {b: i for i, b in vocab.items()}
        # a dict to find ranks of tuples in O(1) time
        self.merge_rank = {t: i for i, t in enumerate(merges)}

    def encode(self, text: str) -> list[int]:
        text_special_tok = []
        if self.special_tokens:
            split_pat = f"({'|'.join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True))})"
            text_special_tok = re.split(split_pat, text)
        else:
            text_special_tok = [text]

        return self._handle_encode(text_special_tok)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        decoded_bytes = b"".join(self.vocab[i] for i in ids)
        return decoded_bytes.decode(errors="replace")

    def _handle_encode(self, text_special_tok: list[str]) -> list[int]:
        encoded_str = []
        for t in text_special_tok:
            if self.special_tokens and t in self.special_tokens:
                encoded_str.append(self.id[t.encode("utf-8")])
                continue
            for match in re.finditer(self.PAT, t):
                pre_token_bytes = [bytes([b]) for b in match.group().encode("utf-8")]
                while True:
                    # TODO: If length == 2 do the final addition thing here
                    max_priority_tuple = None
                    max_rank = float("inf")
                    for i in range(len(pre_token_bytes) - 1):
                        if self.merge_rank.get((pre_token_bytes[i], pre_token_bytes[i + 1]), float("inf")) < max_rank:
                            max_priority_tuple = (pre_token_bytes[i], pre_token_bytes[i + 1])
                            max_rank = self.merge_rank.get((pre_token_bytes[i], pre_token_bytes[i + 1]))
                    if max_priority_tuple == None:
                        break

                    new_pre_token_bytes = []
                    i = 0
                    while i < len(pre_token_bytes):
                        if (
                            i < len(pre_token_bytes) - 1
                            and (pre_token_bytes[i] == max_priority_tuple[0])
                            and (pre_token_bytes[i + 1] == max_priority_tuple[1])
                        ):
                            new_pre_token_bytes.append(pre_token_bytes[i] + pre_token_bytes[i + 1])
                            i += 2
                        else:
                            new_pre_token_bytes.append(pre_token_bytes[i])
                            i += 1
                    pre_token_bytes = new_pre_token_bytes

                for i in range(len(pre_token_bytes)):
                    encoded_str.append(self.id[pre_token_bytes[i]])
        return encoded_str
