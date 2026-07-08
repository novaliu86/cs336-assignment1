

from collections.abc import Iterable, Iterator
import pickle
import regex as re
import sys
import os
from functools import lru_cache
from pathlib import Path
import time
import numpy as np

from cs336_basics.pretokenization import PAT

# MAX_RANK = sys.maxsize

# class Merger:
#     def _get_token_rank(self, start: int, end: int) -> int:
#         if end < 0:
#             end = self.num_bytes

#         if end > self.num_bytes:
#             return MAX_RANK

#         token = self.pretoken[start:end]
#         if token in self.rank_by_bytes:
#             return self.rank_by_bytes[token]
#         return MAX_RANK

#     def __init__(self, pretoken_str: str, rank_by_bytes: dict[bytes, int]):
#         self.rank_by_bytes = rank_by_bytes
#         self.pretoken: bytes = pretoken_str.encode("utf-8")
#         self.num_bytes = len(self.pretoken)
#         assert self.num_bytes > 0, "pretoken has zero bytes"

#         self.linked_list: list[list[int, int]] = [None] * self.num_bytes
#         for i in range(self.num_bytes):
#             next_index = i + 1
#             if next_index == self.num_bytes:
#                 next_index = -1

#             rank = self._get_token_rank(i, i + 2)
#             self.linked_list[i] = [next_index, rank]

#     def _get_min_rank(self) -> int:
#         result = MAX_RANK

#         current_index = 0
#         while (current_index >= 0):
#             [next_index, rank] = self.linked_list[current_index]
#             result = min(result, rank)
#             current_index = next_index

#         return result

#     def _merge(self, target_rank: int):
#         current_index = 0
#         prev_index = -1
#         while (current_index >= 0):
#             current_index = 0 if prev_index < 0 else self.linked_list[prev_index][0]
#             current = self.linked_list[current_index]

#             if current[0] < 0:
#                 current[1] = MAX_RANK
#                 break

#             if current[1] != target_rank:
#                 prev_index = current_index
#                 continue

#             # remove `next` node from double linked list
#             next = self.linked_list[current[0]]
#             next2_index = next[0]
#             if next2_index >= 0:
#                 next2 = self.linked_list[next2_index]
#                 next3_index = next2[0]
#             else:
#                 next3_index = -1
#             current[0] = next2_index

#             current[1] = self._get_token_rank(current_index, next3_index)

#             if prev_index >= 0:
#                 prev = self.linked_list[prev_index]
#                 prev[1] = self._get_token_rank(prev_index, next2_index)

#             prev_index = current_index

#     def merge(self):

#         # print(f"pretoken: {self.pretoken}")

#         while(True):
#             # print(f"self.linked_list: {self.linked_list}")
#             rank = self._get_min_rank()

#             # print(f"next rank: {rank}")
#             if rank == MAX_RANK:
#                 break
#             self._merge(rank)

#     def to_ids(self) -> list[int]:
#         ids = []

#         current_index = 0
#         while (current_index >= 0):
#             [next_index, rank] = self.linked_list[current_index]

#             token = self.pretoken[current_index:(next_index if next_index >= 0 else self.num_bytes)]
#             assert token in self.rank_by_bytes, f"token {token} missing from vocab"
#             ids.append(self.rank_by_bytes[token])
#             current_index = next_index

#         return ids


MAX_TOKEN_ID = sys.maxsize
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

    # Linked-list based tokenizer, which is ~50% slower in our measurement
    # def _encode_pretoken(self, pretoken: str) -> list[int]:
    #     merger = Merger(pretoken, self.token_id_by_bytes)
    #     merger.merge()
    #     return merger.to_ids()

    @lru_cache(maxsize=100000)
    def _encode_pretoken(self, pretoken: str) -> list[int]:
        tokens = [bytes([b]) for b in pretoken.encode("utf-8")]

        while (True):
            merge_index = -1

            merge_id = MAX_TOKEN_ID
            for i in range(len(tokens) - 1):
                new_merge_id = self.token_id_by_bytes.get(tokens[i] + tokens[i+1], MAX_TOKEN_ID)
                if (new_merge_id < merge_id):
                    merge_index = i
                    merge_id = new_merge_id

            if merge_index >= 0:
                tokens[merge_index] = tokens[merge_index] + tokens[merge_index+1]
                del tokens[merge_index+1]
            else:
                break

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
        text_bytes_list = [None] * len(ids)
        for i in range(len(ids)):
            text_bytes_list[i] = self.token_bytes_by_id[ids[i]]
        text_bytes = b"".join(text_bytes_list)
        return text_bytes.decode("utf-8", errors="replace")


def tokenize(
    input_file_path_str: str,
    tokenizer_path_str: str,
    special_tokens: list[str],
):
    output_file_path_str: str = f"{input_file_path_str}-tokens.pkl"

    tokenizer = Tokenizer.from_files(tokenizer_path_str, special_tokens=special_tokens)

    initiated_time = time.perf_counter()

    input_file_path = Path(input_file_path_str)
    token_ids = []
    with open(input_file_path) as f:
        for ids in tokenizer.encode_iterable(f):
            token_ids.append(ids)

    tokenized_time = time.perf_counter()
    print(f"Time to tokenize file {input_file_path}: {(tokenized_time - initiated_time):.6} seconds")

    # Get size in bytes
    file_size_mb = input_file_path.stat().st_size / 1000000
    print(f"File size: {file_size_mb:.4} MB")
    print(f"Number of tokens: {len(token_ids)}")
    print(f"Compression ratio: {(file_size_mb * 1000000 / len(token_ids)):.4} bytes/token")
    print(f"Throuput: {(file_size_mb / (tokenized_time - initiated_time)):.4} MB/second")

    with open(output_file_path_str, "wb") as f:
        pickle.dump(np.array(token_ids, dtype=np.uint16), f)
    print(f"Dumped token ids to {output_file_path_str}")

    verify(input_file_path_str, output_file_path_str, tokenizer)


def verify(
    input_file_path_str: str,
    output_file_path_str: str,
    tokenizer: Tokenizer,
):
    decode_file_path_str = "data/temp.txt"
    if os.path.exists(decode_file_path_str):
        os.remove(decode_file_path_str)

    with open(output_file_path_str, "rb") as f:
        uint16_token_ids = pickle.load(f)
    # print("Loaded all token ids")
    with open(decode_file_path_str, "w+", encoding="utf-8") as f:
        step_size = 10000000
        for start in range(0, len(uint16_token_ids), step_size):
            end = min(start + step_size, len(uint16_token_ids))
            f.write(tokenizer.decode(uint16_token_ids[start:end].tolist()))
    print(f"Decoded token ids to {decode_file_path_str}")

    import filecmp
    are_identical = filecmp.cmp(input_file_path_str, decode_file_path_str, shallow=False)

    if are_identical:
        print("Decoded file matches original file perfectly.")
        os.remove(decode_file_path_str)
    else:
        print(f"Decoded file is different from original file {input_file_path_str}!")


if __name__ == "__main__":
    tokenize(
        "data/owt_valid.txt",
        "data/BPE-owt.pkl",
        ["<|endoftext|>"]
    )
