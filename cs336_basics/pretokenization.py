import math
import os
import regex as re
from collections import Counter
from typing import BinaryIO
import multiprocessing


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token,
                      bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def process_chunk(
    input_path: str | os.PathLike,
    start: int,
    end: int,
    special_tokens: list[str],
) -> Counter[str]:
    doc_splilt_re = "|".join([re.escape(special_token)
                              for special_token in special_tokens])
    with open(input_path, "rb") as data_file:
        data_file.seek(start)
        chunk = data_file.read(
            end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        docs = re.split(doc_splilt_re, chunk)

        chunk_counter: Counter[str] = Counter()
        for doc in docs:
            chunk_counter.update(re.findall(PAT, doc))
        return chunk_counter


def process_batch(chunk_params: list, batch_index: int) -> Counter[str]:
    counter: Counter[str] = Counter()
    for chunk_index in range(len(chunk_params)):
        [input_path, start, end, special_tokens] = chunk_params[chunk_index]
        chunk_counter = process_chunk(input_path, start, end, special_tokens)
        counter.update(chunk_counter)
        # print(f"  Processed chunk {batch_index}.{chunk_index}")

    return counter


def split_into_n_batches_rigid(data, num_batches):
    batch_size = math.ceil(len(data) / num_batches)
    result = []
    for i in range(0, len(data), batch_size):
        result.append((data[i: i + batch_size], int(i / batch_size)))
    return result


def count_pretokens(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_chunks: int,
) -> Counter[str]:
    assert len(special_tokens) >= 1, "no special tokens provided"

    with open(input_path, "rb") as data_file:
        chunk_split_special_token = special_tokens[0].encode("utf-8")
        boundaries = find_chunk_boundaries(
            data_file, num_chunks, chunk_split_special_token)
    print(f"Detected {len(boundaries) - 1} chunks.")

    num_cores = multiprocessing.cpu_count()
    print(f"Starting parallel processing using {num_cores} CPU cores...")
    batch_params = split_into_n_batches_rigid([(input_path, start, end, special_tokens)
                                              for start, end in zip(boundaries[:-1], boundaries[1:])], num_cores)
    with multiprocessing.Pool() as pool:
        batch_counters = pool.starmap(process_batch, batch_params)

    counter: Counter[str] = Counter()
    for batch_counter in batch_counters:
        counter.update(batch_counter)
    return counter
