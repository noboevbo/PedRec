from typing import Optional

import numpy as np


def set_or_vstack(a: np.ndarray, b: np.ndarray, expand_dim_on_set: bool = True):
    """
    If a exists it vstacks b, if not it sets a to b.
    :param a:
    :param b:
    :return:
    """
    if expand_dim_on_set:
        b = np.expand_dims(b, axis=0)
    if a is None:
        return b
    else:
        return np.vstack((a, b))


def fill_array(array: np.ndarray, array_size: int, fill_value: int):
    if array.shape[0] < array_size:
        shape = [array_size - array.shape[0]]
        for shape_idx in range(1, len(array.shape)):
            shape.append(array.shape[shape_idx])
        fill_array = np.ones(tuple(shape)) * fill_value
        return np.concatenate((array, fill_array))
    else:
        return array


def split_numpy_array(input_array: np.ndarray, split_size: int, fill_value: int = None):
    """
    Currently only supports split on axis=0
    """
    output_splits: np.ndarray = None
    for i in range(0, input_array.shape[0] // split_size):
        idx_from = i * split_size
        idx_to = idx_from + split_size
        output_splits = set_or_vstack(output_splits, input_array[idx_from:idx_to])
    if fill_value is not None:
        rest = input_array.shape[0] % split_size
        if rest != 0:
            rest_array = fill_array(input_array[-rest:], split_size, fill_value)
            output_splits = set_or_vstack(output_splits, rest_array)
    return output_splits


def split_numpy_array_stepwise(input_array: np.ndarray, split_size: int, step_size: int,
                               fill_value: int = None) -> np.ndarray:
    """
    Currently only supports split on axis=0
    """
    assert fill_value is not None or input_array.shape[0] >= split_size, \
        "Input list of length '{}' is less than split size '{}' and no fill_value is given".format(input_array.shape[0],
                                                                                                   split_size)
    # TODO: Create a handler which checks for splits which contain only zeros or so and removes them
    output_splits: np.ndarray = None
    if input_array.shape[0] < split_size:
        filled_array = fill_array(input_array, split_size, fill_value)
        return np.expand_dims(filled_array, axis=0)

    for input_list_index in range(0, input_array.shape[0] - (split_size - 1), step_size):  # TODO: is this -1 correct?
        output_split = input_array[input_list_index:input_list_index + split_size]
        output_splits = set_or_vstack(output_splits, output_split)
    return output_splits

def create_meshgrid_np(
        x: np.ndarray,
        normalized_coordinates: Optional[bool]) -> np.ndarray:
    # assert len(x.shape) == 4, x.shape
    # _, _, height, width = x.shape
    height, width = x.shape
    _dtype = x.dtype
    if normalized_coordinates:
        xs = np.linspace(0.0, 1.0, width, dtype=_dtype)
        ys = np.linspace(0.0, 1.0, height, dtype=_dtype)
    else:
        xs = np.linspace(0, width - 1, width, dtype=_dtype)
        ys = np.linspace(0, height - 1, height, dtype=_dtype)
    return np.meshgrid(ys, xs)
