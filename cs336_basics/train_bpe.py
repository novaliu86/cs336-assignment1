
import os
from collections import Counter

from cs336_basics.pretokenization import count_pretokens


class MergeManager:
    def __init__(self, pretoken_counter: Counter[str]):

        self.pretoken_elements_and_counts: list[tuple[list[bytes], int]] = []
        for pretoken, count in pretoken_counter.items():
            elements = [bytes([b]) for b in pretoken.encode("utf-8")]
            self.pretoken_elements_and_counts.append((elements, count))

        self.pairs: dict[tuple[bytes, bytes], list[int, list[int]]] = {}

        for i in range(len(self.pretoken_elements_and_counts)):
            (elements, pretoken_count) = self.pretoken_elements_and_counts[i]
            for j in range(len(elements) - 1):
                self._add_pair_count((elements[j], elements[j + 1]), pretoken_count, i)

        # print(self.pretoken_elements_and_counts)
        # print(self.pairs)

    def _add_pair_count(self, pair: tuple[bytes, bytes], pretoken_count: int, pretoken_index: int):
        pair_entry = self.pairs.setdefault(pair, [0, []])
        pair_entry[0] += pretoken_count
        if (len(pair_entry[1]) == 0 or pair_entry[1][-1] != pretoken_index):
            pair_entry[1].append(pretoken_index)
    
    def _reduce_pair_count(self, pair: tuple[bytes, bytes], pretoken_count: int):
        assert pair in self.pairs, "missing pair"
        pair_entry = self.pairs[pair]
        pair_entry[0] -= pretoken_count
        if (pair_entry[0] == 0):
            del self.pairs[pair]

    def has_next(self) -> bool:
        return len(self.pairs) > 0

    def next(self) -> tuple[tuple[bytes, bytes], bytes]:
        (merge, [count, pretoken_indices]) = ((b"", b""), [0, []])
        for p, e in self.pairs.items():
            if e[0] > count or (e[0] == count and p > merge):
                merge = p
                count = e[0]
                pretoken_indices = e[1]

        (a, b) = merge
        token = a + b
        # print("found max:", merge, count, pretoken_indices)

        for pretoken_index in pretoken_indices:
            (pretoken_elements,
             pretoken_count) = self.pretoken_elements_and_counts[pretoken_index]
            # for j in range(len(pretoken_elements) - 1):
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
                    self._add_pair_count((pretoken_elements[j-1], token), pretoken_count, pretoken_index)
                    self._reduce_pair_count((pretoken_elements[j-1], a), pretoken_count)
                if (j + 1 < len(pretoken_elements)):
                    self._add_pair_count((token, pretoken_elements[j+1]), pretoken_count, pretoken_index)
                    self._reduce_pair_count((b, pretoken_elements[j+1]), pretoken_count)

        # print(self.pretoken_elements_and_counts)
        del self.pairs[merge]
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
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    pretoken_counter = count_pretokens(input_path, special_tokens)
    merge_manager = MergeManager(pretoken_counter)

    return build_vocab(merge_manager, vocab_size, special_tokens)


if __name__ == "__main__":
    # pretoken_counter = Counter()
    # pretoken_counter.update(["low", "low", "low", "low", "low", "lower", "lower", "widest", "widest", "widest", "newest", "newest", "newest", "newest", "newest", "newest"])

    # merge_manager = MergeManager(pretoken_counter)
    # (vocab, merges) = build_vocab(merge_manager, 256 + 9, [])
    # print(vocab)
    # print(merges)

    import time

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