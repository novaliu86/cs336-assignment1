

from collections.abc import Iterable, Iterator
import pickle
import regex as re
import sys
from functools import lru_cache

from cs336_basics.pretokenization import PAT

MAX_RANK = sys.maxsize

class Merger:
    def _get_token_rank(self, start: int, end: int) -> int:
        if end < 0:
            end = len(self.pretoken)

        if end > len(self.pretoken):
            return MAX_RANK

        token = self.pretoken[start:end]
        if token in self.rank_by_bytes:
            return self.rank_by_bytes[token]
        return MAX_RANK

    def __init__(self, pretoken_str: str, rank_by_bytes: dict[bytes, int]):
        self.rank_by_bytes = rank_by_bytes
        self.pretoken: bytes = pretoken_str.encode("utf-8")
        num_bytes = len(self.pretoken)
        assert num_bytes > 0, "pretoken has zero bytes"

        self.linked_list: list[list[int, int, int]] = [None] * num_bytes
        for i in range(num_bytes):
            prev_index = i - 1

            next_index = i + 1
            if next_index == num_bytes:
                next_index = -1

            rank = self._get_token_rank(i, i + 2)
            self.linked_list[i] = [prev_index, next_index, rank]

    def _get_min_rank(self) -> int:
        result = MAX_RANK

        current_index = 0
        while (current_index >= 0):
            [_, next_index, rank] = self.linked_list[current_index]
            result = min(result, rank)
            current_index = next_index

        return result

    def _merge(self, target_rank: int):
        current_index = 0
        while (current_index >= 0):
            current = self.linked_list[current_index]

            if current[1] < 0:
                current[2] = MAX_RANK
                break

            if current[2] != target_rank:
                current_index = current[1]
                continue

            # remove `next` node from double linked list
            next = self.linked_list[current[1]]
            next2_index = next[1]
            if next2_index >= 0:
                next2 = self.linked_list[next2_index]
                next2[0] = current_index
                next3_index = next2[1]
            else:
                next3_index = -1
            current[1] = next2_index

            current[2] = self._get_token_rank(current_index, next3_index)

            prev_index = current[0]
            if prev_index >= 0:
                prev = self.linked_list[prev_index]
                prev[2] = self._get_token_rank(prev_index, next2_index)

            current_index = current[1]

    def merge(self):

        # print(f"pretoken: {self.pretoken}")

        while(True):
            # print(f"self.linked_list: {self.linked_list}")
            rank = self._get_min_rank()

            # print(f"next rank: {rank}")
            if rank == MAX_RANK:
                break
            self._merge(rank)

    def to_ids(self) -> list[int]:
        ids = []

        current_index = 0
        while (current_index >= 0):
            [_, next_index, rank] = self.linked_list[current_index]

            token = self.pretoken[current_index:(next_index if next_index >= 0 else len(self.pretoken))]
            assert token in self.rank_by_bytes, f"token {token} missing from vocab"
            ids.append(self.rank_by_bytes[token])
            current_index = next_index

        return ids


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.token_bytes_by_id: dict[int, bytes] = vocab
        self.token_id_by_bytes: dict[bytes, int] = {
            bytes: id for id, bytes in vocab.items()}

        # for (a, b) in merges:
        #     token = a + b
        #     assert token in self.token_id_by_bytes, f"missing token {token}"
        # self.pairs: set[tuple[bytes, bytes]] = set(merges)

        special_tokens = [] if special_tokens is None else special_tokens
        for special_token in special_tokens:
            assert special_token.encode(
                "utf-8") in self.token_id_by_bytes, f"missing special token {special_token}"
        special_tokens.sort(key=lambda x: -len(x))
        self.special_tokens: list[str] = special_tokens
        self.doc_splilt_re = "|".join([re.escape(special_token)
                                       for special_token in special_tokens])

    @classmethod
    def from_files(cls, vocab_merges_filepath, special_tokens=None):
        with open(vocab_merges_filepath, "rb") as f:
            (vocab, merges) = pickle.load(f)
            return cls(vocab, merges, special_tokens)

    def _encode_special_tokens(self, text: str, start: int) -> tuple[list[int], int]:
        end = start
        token_ids = []
        while (end < len(text)):
            has_special_token = False
            for special_token in self.special_tokens:
                if text[end:].startswith(special_token):
                    end += len(special_token)
                    token_ids.append(
                        self.token_id_by_bytes[special_token.encode("utf-8")])
                    has_special_token = True
                    break
            if not has_special_token:
                break
        return (token_ids, end)

    # @lru_cache(maxsize=100000)
    # def _encode_pretoken(self, pretoken: str) -> list[int]:
    #     merger = Merger(pretoken, self.token_id_by_bytes)
    #     merger.merge()
    #     return merger.to_ids()

    # @lru_cache(maxsize=100000)
    def _encode_pretoken(self, pretoken: str) -> list[int]:
        tokens = [bytes([b]) for b in pretoken.encode("utf-8")]

        while (True):
            merge_id = -1
            merge_index = -1
            for i in range(len(tokens) - 1):
                merged = tokens[i] + tokens[i+1]
                if merged in self.token_id_by_bytes:
                    new_merge_id = self.token_id_by_bytes[merged]
                    if (merge_id < 0 or new_merge_id < merge_id):
                        merge_id = new_merge_id
                        merge_index = i
            if merge_index < 0:
                break
            else:
                tokens[merge_index] = tokens[merge_index] + \
                    tokens[merge_index+1]
                del tokens[merge_index+1]

        return [self.token_id_by_bytes[token] for token in tokens]

    def _encode(self, text: str) -> tuple[list[int], int]:
        if self.doc_splilt_re == "":
            docs = [text]
        else:
            docs = re.split(self.doc_splilt_re, text)

        (token_ids, special_tokens_end) = self._encode_special_tokens(text, 0)
        last_pretoken_start = len(token_ids)  # start index of last pretoken in token_ids list

        for doc in docs:
            pretokens = re.findall(PAT, doc)
            for j in range(len(pretokens)):
                if j == len(pretokens) - 1:
                    last_pretoken_start = len(token_ids)
                token_ids.extend(self._encode_pretoken(pretokens[j]))

            (special_token_ids, special_tokens_end) = self._encode_special_tokens(text, special_tokens_end + len(doc))
            token_ids.extend(special_token_ids)
            if (len(special_token_ids) > 0):
                last_pretoken_start = len(token_ids)
        return (token_ids, last_pretoken_start)

    # to prevent memory leak caused by lru cache
    def __del__(self):
        try:
            self._encode_special_tokens.cache_clear()
        except AttributeError:
            pass

    def encode(self, text: str) -> list[int]:
        (token_ids, _) = self._encode(text)
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        left_over_text = ""
        for new_text in iterable:
            (token_ids, last_pretoken_start) = self._encode(left_over_text + new_text)
            left_over_text = self.decode(token_ids[last_pretoken_start:])
            yield from token_ids[:last_pretoken_start]
        yield from token_ids[last_pretoken_start:]

    def decode(self, ids: list[int]) -> str:
        text_bytes = b""
        for id in ids:
            text_bytes += self.token_bytes_by_id[id]
        return text_bytes.decode("utf-8", errors="replace")

if __name__ == "__main__":
    from pathlib import Path
    import time

    file_to_tokenize = Path("data/owt_valid.txt")
    # file_to_tokenize = Path("data/TinyStoriesV2-GPT4-valid.txt")

    start_time = time.perf_counter()

    tokenizer = Tokenizer.from_files("data/BPE-TinyStoriesV2-GPT4.pkl", special_tokens=["<|endoftext|>"])

    initiated_time = time.perf_counter()

    print(f"Time to initialize tokenizer: {(initiated_time - start_time):.6} seconds")


    token_ids = []
    with open(file_to_tokenize) as f:
        for ids in tokenizer.encode_iterable(f):
            token_ids.append(ids)

    tokenized_time = time.perf_counter()
    print(f"Time to tokenize file {file_to_tokenize}: {(tokenized_time - initiated_time):.6} seconds")

    # Get size in bytes
    file_size_mb = file_to_tokenize.stat().st_size / 1000000
    print(f"File size: {file_size_mb:.4} MB")
    print(f"Number of tokens: {len(token_ids)}")
    print(f"Compression ratio: {(file_size_mb * 1000000 / len(token_ids)):.4} bytes/token")
    print(f"Throuput: {(file_size_mb / (tokenized_time - initiated_time)):.4} MB/second")