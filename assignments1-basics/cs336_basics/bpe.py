from typing import BinaryIO
import regex as re

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process

from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> (dict[int, bytes], list[tuple[bytes, ...]]):
    # pretokenize
    # vocab and freq list
    chunks: list[str] = chunks_array(input_path)
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges = []
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")
    word_counts = {}
    for chunk in chunks:
        chunk_no_special_tok = None
        # removing special characters
        if special_tokens:
            split_pat = "|".join(re.escape(t) for t in special_tokens)
            chunk_no_special_tok = re.split(split_pat, chunk)
        else:
            chunk_no_special_tok = [chunk]
        # pretokenize
        for c in chunk_no_special_tok:
            for match in re.finditer(PAT, c):
                tup = tuple(bytes([b]) for b in match.group().encode("utf-8"))
                word_counts[tup] = word_counts.get(tup, 0) + 1
    while len(vocab) < vocab_size:
        pair_counts = {}
        for t, c in word_counts.items():
            for i in range(len(t) - 1):
                pair_counts[(t[i], t[i + 1])] = pair_counts.get((t[i], t[i + 1]), 0) + c
                # hopping two steps
        if len(pair_counts) == 0:
            break
        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))
        merges.append((best_pair))
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        new_word_counts = {}
        for t, c in word_counts.items():
            i = 0
            tn = []
            while i<len(t):
                if i<len(t) - 1 and t[i]==best_pair[0] and t[i+1]==best_pair[1]:
                    tn.append(best_pair[0]+best_pair[1])
                    i+=2
                else:
                    tn.append(t[i])
                    i+=1
            new_word_counts[tuple(tn)] = c
        word_counts = new_word_counts
    return vocab, merges


# def freq_list(chunk : str) -> dict[int, bytes], list[tuple[bytes,...]]:


def chunks_array(input_path: str) -> list[str]:
    with open(
        input_path,
        "rb",
    ) as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
        return chunks


# def pretokenize
