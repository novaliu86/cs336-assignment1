
import os
from pathlib import Path
from collections import Counter
from sortedcontainers import SortedList
import time
import pickle

from cs336_basics.pretokenization import count_pretokens


class PairMap:
    def __init__(self):
        self._lookup: dict[tuple[bytes, bytes], list[int, list[int]]] = {}
        self._sorted_items = SortedList(
            key=lambda item: (item[1][0], item[0]))

    def add_pair_count(self, pair: tuple[bytes, bytes], pretoken_count: int, pretoken_index: int):
        if pair in self._lookup:
            value = self._lookup[pair]
            self._sorted_items.remove((pair, value))
        else:
            value = [0, []]
            self._lookup[pair] = value

        value[0] += pretoken_count
        if (len(value[1]) == 0 or value[1][-1] != pretoken_index):
            value[1].append(pretoken_index)

        self._lookup[pair] = value
        self._sorted_items.add((pair, value))

    def reduce_pair_count(self, pair: tuple[bytes, bytes], pretoken_count: int):
        assert pair in self._lookup, "missing pair I"
        value = self._lookup[pair]
        self._sorted_items.remove((pair, value))

        if (value[0] == pretoken_count):
            self._lookup.pop(pair)
            return

        value[0] -= pretoken_count

        self._sorted_items.add((pair, value))

    def get_next_pair(self) -> tuple[tuple[bytes, bytes], int, list[int]]:
        (pair, [count, pretoken_indices]) = self._sorted_items[-1]
        return (pair, count, pretoken_indices)

    def remove_next_pair(self, pair: tuple[bytes, bytes]):
        assert pair in self._lookup, "missing pair II"
        value = self._lookup.pop(pair)
        self._sorted_items.remove((pair, value))

    def has_next_pair(self) -> bool:
        return len(self._sorted_items) > 0


class MergeManager:
    def __init__(self, pretoken_counter: Counter[str]):

        self.pretoken_elements_and_counts: list[tuple[list[bytes], int]] = []
        for pretoken, count in pretoken_counter.items():
            elements = [bytes([b]) for b in pretoken.encode("utf-8")]
            self.pretoken_elements_and_counts.append((elements, count))

        self.pairs: PairMap = PairMap()

        for i in range(len(self.pretoken_elements_and_counts)):
            (elements, pretoken_count) = self.pretoken_elements_and_counts[i]
            for j in range(len(elements) - 1):
                self.pairs.add_pair_count(
                    (elements[j], elements[j + 1]), pretoken_count, i)

    def has_next(self) -> bool:
        return self.pairs.has_next_pair()

    def next(self) -> tuple[tuple[bytes, bytes], bytes]:
        (merge, count, pretoken_indices) = self.pairs.get_next_pair()

        (a, b) = merge
        token = a + b
        # print("found max:", merge, count, pretoken_indices)

        for pretoken_index in pretoken_indices:
            (pretoken_elements,
             pretoken_count) = self.pretoken_elements_and_counts[pretoken_index]
            j = -1
            while (True):
                j += 1
                if j+1 >= len(pretoken_elements):
                    break

                if (pretoken_elements[j] != a or pretoken_elements[j+1] != b):
                    continue
                # merge and reduce len(pretoken_elements)
                # 1. edit pretoken_elements
                pretoken_elements[j] = token
                del pretoken_elements[j+1]
                # 2. add new pairs: (x, a, b, y) => (x, ab, y)
                if (j > 0):
                    self.pairs.add_pair_count(
                        (pretoken_elements[j-1], token), pretoken_count, pretoken_index)
                    self.pairs.reduce_pair_count(
                        (pretoken_elements[j-1], a), pretoken_count)
                if (j + 1 < len(pretoken_elements)):
                    self.pairs.add_pair_count(
                        (token, pretoken_elements[j+1]), pretoken_count, pretoken_index)
                    self.pairs.reduce_pair_count(
                        (b, pretoken_elements[j+1]), pretoken_count)

        # print(self.pretoken_elements_and_counts)
        self.pairs.remove_next_pair(merge)
        # print(self.pairs._lookup)
        # print(self.pairs._sorted_items)
        return (merge, token)


def build_vocab(
    merge_manager: MergeManager,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    for token in [special_token.encode("utf-8") for special_token in special_tokens] + [bytes([b]) for b in range(256)]:
        vocab[len(vocab)] = token
        if len(vocab) >= vocab_size:
            return (vocab, merges)

    while len(vocab) < vocab_size:
        if not merge_manager.has_next():
            break

        (merge, token) = merge_manager.next()
        vocab[len(vocab)] = token
        merges.append(merge)

    return (vocab, merges)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    pretoken_counter_cache_path = Path(input_path + ".pkl")

    if pretoken_counter_cache_path.is_file():
        with open(pretoken_counter_cache_path, "rb") as f:
            pretoken_counter = pickle.load(f)
        print(
            f"Loaded cached pretoken counter from {pretoken_counter_cache_path}")
        for item, count in pretoken_counter.most_common(10):
            print(f"  {item}: {count}")
    else:
        pretoken_counter = count_pretokens(input_path, special_tokens, 100)
        with open(pretoken_counter_cache_path, "wb") as f:
            pickle.dump(pretoken_counter, f)
        print(f"Counted pretokens and dumped to {pretoken_counter_cache_path}")

    merge_manager = MergeManager(pretoken_counter)

    return build_vocab(merge_manager, vocab_size, special_tokens)


def train_bpe_tinystories():
    start_time = time.perf_counter()

    # --- Code you want to measure starts here ---
    (vocab, merges) = train_bpe(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"]
    )
    # --- Code you want to measure ends here ---

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time:.6f} seconds")

    with open("data/BPE-TinyStoriesV2-GPT4.pkl", "wb") as f:
        pickle.dump((vocab, merges), f)


def train_bpe_expts_owt():
    start_time = time.perf_counter()

    # --- Code you want to measure starts here ---
    (vocab, merges) = train_bpe(
        input_path="data/owt_train.txt",
        vocab_size=32000,
        special_tokens=["<|endoftext|>"]
    )
    # --- Code you want to measure ends here ---

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time:.6f} seconds")

    with open("data/BPE-owt.pkl", "wb") as f:
        pickle.dump((vocab, merges), f)


def exam_bpe_tinystories():
    with open("data/BPE-TinyStoriesV2-GPT4.pkl", "rb") as f:
        (vocab, _) = pickle.load(f)

    # Sorts the items in descending order by length and grabs the first 10
    longest_10_items = sorted(
        vocab.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    print(longest_10_items)


def exam_bpe_owt():
    with open("data/BPE-owt.pkl", "rb") as f:
        (vocab, _) = pickle.load(f)

    # Sorts the items in descending order by length and grabs the first 10
    longest_10_items = sorted(
        vocab.items(), key=lambda x: len(x[1]), reverse=True)[:100]
    print(longest_10_items)


def test_bpe_minimum_case():
    pretoken_counter = Counter()
    pretoken_counter.update(["low", "low", "low", "low", "low", "lower", "lower", "widest",
                            "widest", "widest", "newest", "newest", "newest", "newest", "newest", "newest"])

    merge_manager = MergeManager(pretoken_counter)
    (vocab, merges) = build_vocab(merge_manager, 256 + 9, [])
    print(vocab)
    print(merges)


if __name__ == "__main__":
    # test_bpe_minimum_case()
    
    # train_bpe_tinystories()
    # exam_bpe_tinystories()

    train_bpe_expts_owt()
    exam_bpe_owt()
