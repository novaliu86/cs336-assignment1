

from collections.abc import Iterable, Iterator
import pickle
import regex as re

from cs336_basics.pretokenization import PAT


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
        self.special_tokens: set[str] = set(special_tokens)
        self.doc_splilt_re = "|".join([re.escape(special_token)
                                       for special_token in special_tokens])

    def from_files(cls, vocab_merges_filepath, special_tokens=None):
        with open(vocab_merges_filepath, "rb") as f:
            (vocab, merges) = pickle.load(f)
            return Tokenizer(vocab, merges, special_tokens)

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

    def encode(self, text: str) -> list[int]:
        if self.doc_splilt_re == "":
            docs = [text]
        else:
            docs = re.split(self.doc_splilt_re, text)

        (tokens, i) = self._encode_special_tokens(text, 0)
        for doc in docs:
            if (doc == ""):
                continue
            pretokens = re.findall(PAT, doc)
            for pretoken in pretokens:
                tokens.extend(self._encode_pretoken(pretoken))

            (special_tokens, i) = self._encode_special_tokens(text, i + len(doc))
            tokens.extend(special_tokens)
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            token_ids = self.encode(text)
            yield from token_ids

    def decode(self, ids: list[int]) -> str:
        text_bytes = b""
        for id in ids:
            text_bytes += self.token_bytes_by_id[id]
        return text_bytes.decode("utf-8", errors="replace")
