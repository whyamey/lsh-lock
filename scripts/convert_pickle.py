#!/usr/bin/env python3
import pickle
import numpy as np
import struct
import sys


def convert_pickle_to_bincode(pickle_path: str, bincode_path: str):
    """
    Converts a pickled numpy array from Original CompFE to a Rust-compatible bincode
    format that matches the RandomIndices struct for the Rust implementation.
    """
    with open(pickle_path, "rb") as f:
        positions = pickle.load(f)

    positions = np.array(positions)

    positions = positions.astype(np.uint64)

    with open(bincode_path, "wb") as f:
        f.write(struct.pack("<Q", len(positions)))

        for arr in positions:
            f.write(struct.pack("<Q", len(arr)))
            for val in arr:
                f.write(struct.pack("<Q", val))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_pickle.py input.pkl output.bin")
        sys.exit(1)

    convert_pickle_to_bincode(sys.argv[1], sys.argv[2])
