import numpy as np


def chunk_data(array, chunk_size):
    array = np.array_split(array, len(array)/chunk_size)
    return np.array(array)
