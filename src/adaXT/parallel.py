from functools import partial
import multiprocessing
from multiprocessing import cpu_count
from itertools import starmap
from typing import Any, Callable, Iterable

from multiprocessing import RawArray
import numpy as np
import ctypes


def shared_numpy_array(array) -> np.ndarray:
    if array.ndim == 2:
        row, col = array.shape
        shared_array = RawArray(ctypes.c_double, (row * col))
        shared_array_np = np.ndarray(
            shape=(row, col), dtype=np.double, buffer=shared_array
        )
    elif array.ndim == 1:
        row = array.shape[0]
        shared_array = RawArray(ctypes.c_double, row)
        shared_array_np = np.ndarray(
            shape=row, dtype=np.double, buffer=shared_array)
    else:
        raise ValueError("Array is neither 1 dimensional nor 2 dimensional")
    np.copyto(shared_array_np, array)
    return shared_array_np


class ParallelModel:
    """
    Class used to parallelize
    """

    def __init__(
        self,
        n_jobs: int = -1,
    ):
        """
        Parameters
        ----------
        n_jobs : int, default=Number of cpu cores
            The number of processes used to fit, and predict for the forest, -1
            uses all available proccesors
        random_state: int
            Used for deterministic seeding of the tree
        """
        self.ctx = multiprocessing.get_context("fork")
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()

    def async_map(self, function: Callable, map_input: Any, **kwargs) -> Any:
        partial_func = partial(function, **kwargs)
        if self.n_jobs == 1:
            ret = list(map(partial_func, map_input))
        else:
            with self.ctx.Pool(self.n_jobs) as p:
                promise = p.map_async(partial_func, map_input)
                ret = promise.get()
        return ret

    def map(self, function: Callable, map_input: Any, **kwargs) -> Any:
        partial_func = partial(function, **kwargs)
        if self.n_jobs == 1:
            ret = list(map(partial_func, map_input))
        else:
            with self.ctx.Pool(self.n_jobs) as p:
                ret = p.map(partial_func, map_input)
        return ret

    def async_starmap(
            self,
            function: Callable,
            map_input: Iterable,
            **kwargs) -> Any:
        partial_func = partial(function, **kwargs)
        if self.n_jobs == 1:
            ret = list(starmap(partial_func, map_input))
        else:
            with self.ctx.Pool(self.n_jobs) as p:
                promise = p.starmap_async(partial_func, map_input)
                ret = promise.get()
        return ret

    def starmap(
            self,
            function: Callable,
            map_input: Iterable,
            **kwargs) -> Any:
        partial_func = partial(function, **kwargs)
        if self.n_jobs == 1:
            ret = list(starmap(partial_func, map_input))
        else:
            with self.ctx.Pool(self.n_jobs) as p:
                ret = p.starmap(partial_func, map_input)
        return ret

    def async_apply(
            self,
            function: Callable,
            n_iterations: int,
            **kwargs) -> Any:
        partial_func = partial(function, **kwargs)
        if self.n_jobs == 1:
            ret = [partial_func() for _ in range(n_iterations)]
        else:
            with self.ctx.Pool(self.n_jobs) as p:
                promise = [p.apply_async(partial_func)
                           for _ in range(n_iterations)]
                ret = [res.get() for res in promise]
        return ret

    def apply(self, function: Callable, n_iterations: int, **kwargs) -> Any:
        partial_func = partial(function, **kwargs)
        if self.n_jobs == 1:
            ret = [partial_func() for _ in range(n_iterations)]
        else:
            with self.ctx.Pool(self.n_jobs) as p:
                ret = [p.apply(partial_func) for _ in range(n_iterations)]
        return ret
