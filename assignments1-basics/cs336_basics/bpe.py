import time

import regex as re

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
    # adding tokens to the vocab
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")
    word_counts = {}
    for chunk in chunks:
        chunk_no_special_tok = None
        # removing special characters
        if special_tokens:
            # this is creating regex for special token
            # re.escape would make <|end|> => <\|end\|>
            # joining these would make a pattern <\|end\|> | <\|start\|>
            # spliting with this pattern would split text at this tokens with them
            # not being a part of the text
            split_pat = "|".join(re.escape(t) for t in special_tokens)
            chunk_no_special_tok = re.split(split_pat, chunk)
        else:
            chunk_no_special_tok = [chunk]
        # pretokenize
        for c in chunk_no_special_tok:
            # finditer will find a word like "hello"
            # encode will convert it to [104, 101, 108, 108, 111]
            # bytes would convert 104 to b'h'
            # word_counts would count the number of times something appears in our corpus
            for match in re.finditer(PAT, c):
                tup = tuple(bytes([b]) for b in match.group().encode("utf-8"))
                word_counts[tup] = word_counts.get(tup, 0) + 1
    pair_counts = {}
    # filling pair counts
    for t, c in word_counts.items():
        for i in range(len(t) - 1):
            pair_counts[(t[i], t[i + 1])] = pair_counts.get((t[i], t[i + 1]), 0) + c
            # hopping two steps
    while len(vocab) < vocab_size:
        if len(pair_counts) == 0:
            break
        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        # replacing the new pair in word_counts
        # by creating a new word_counts
        # we add existing unchanged tuples as well as changed tuples to word_count
        new_word_counts = {}
        for t, c in word_counts.items():
            i = 0  # index on words
            j = 0  # index on tuples
            # insert = False
            tn = []
            while i < len(t):
                if i < len(t) - 1 and t[i] == best_pair[0] and t[i + 1] == best_pair[1]:
                    tn.append(best_pair[0] + best_pair[1])
                    # insert = True
                    j = len(tn) - 1
                    # FIX: remove counts of a,b where we have the following match
                    if i > 0:
                        pair_counts[(t[i - 1], t[i])] = pair_counts.get((t[i - 1], t[i]), 0) - c
                    if i + 2 < len(t):
                        pair_counts[(t[i + 1], t[i + 2])] = pair_counts.get((t[i + 1], t[i + 2]), 0) - c
                    # NOTE: This adds a,bc and bc,d counts to pair_counts
                    if i + 2 < len(t):
                        pair_counts[(tn[j], t[i + 2])] = pair_counts.get((tn[j], t[i + 2]), 0) + c
                    if i > 0:
                        pair_counts[(tn[j - 1], tn[j])] = pair_counts.get((tn[j - 1], tn[j]), 0) + c
                    i += 2
                else:
                    tn.append(t[i])
                    i += 1
            new_word_counts[tuple(tn)] = c
            # if insert:
            #     for i in range(len(tn) - 1):
            #         pair_counts[(tn[i], tn[i + 1])] = pair_counts.get((tn[i], tn[i + 1]), 0) + c
        pair_counts.pop(best_pair)

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

if __name__ == "__main__":
    start = time.perf_counter()
    train_bpe("tests/fixtures/corpus.en", 500, ["<|endoftext|>"])
    end = time.perf_counter()
    print(end - start)
