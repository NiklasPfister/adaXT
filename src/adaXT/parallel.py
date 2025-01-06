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
        """
        self.ctx = multiprocessing.get_context("fork")
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()

    def async_map(
            self,
            function: Callable,
            map_input: Iterable,
            **kwargs) -> Iterable:
        """
        Asynchronously applies the function to the map_input passing along any
        kwargs given to the function.

        Parameters
        ----------
        function
            Function to apply Asynchronously
        map_input
            Iterable input which can be passed to the function

        Returns
        -------
        Iterable
            Returns the result of running function on all elements of map_input
        """
        partial_func = partial(function, **kwargs)
        if self.n_jobs == 1 or ("__no_parallel" in kwargs):
            ret = list(map(partial_func, map_input))
        else:
            with self.ctx.Pool(self.n_jobs) as p:
                promise = p.map_async(partial_func, map_input)
                ret = promise.get()
        return ret

    def map(
            self,
            function: Callable,
            map_input: Iterable,
            **kwargs) -> Iterable:
        """
        Maps the function with map_input. Similair to async_map, but instead
        guarantees that the first element returned is the result of the first
        map_input. Passes along any kwargs to function.



        Parameters
        ----------
        function
            function to apply
        map_input
            Iterable input which can be passed to the function

        Returns
        -------
        Iterable
            Returns in order the results of function applied to map_input
        """

        partial_func = partial(function, **kwargs)
        if self.n_jobs == 1 or ("__no_parallel" in kwargs):
            ret = list(map(partial_func, map_input))
        else:
            with self.ctx.Pool(self.n_jobs) as p:
                ret = p.map(partial_func, map_input)
        return ret

    def async_starmap(
        self, function: Callable, map_input: Iterable, **kwargs
    ) -> Iterable:
        """
        Asynchronously apply function to map_input, where map_input might be a
        list of tuple elements. Passes along any kwargs to function.


        Parameters
        ----------
        function
            Function to apply to each element of map_input
        map_input
            Iterable input which might be a tuple, that can be passed to
            function

        Returns
        -------
        Iterable
            Returns the result of applying function to each element of map_input
        """
        partial_func = partial(function, **kwargs)
        if self.n_jobs == 1 or ("__no_parallel" in kwargs):
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
        """
        Applies function to each elemetn of map_input but guarantees that
        element i of return value is the result of function applied to element i
        of map_input. Can be a list of tuples as opposed to just map. Passes
        along any kwargs to function.


        Parameters
        ----------
        function
            Function to apply to each element of map_input
        map_input
            Iterable input which might be a tuple, that can be passed to
            function

        Returns
        -------
        Iterable
            Returns the result of applying function to each element of map_input
        """
        partial_func = partial(function, **kwargs)
        if (self.n_jobs == 1) or (
            ("__no_parallel" in kwargs) and kwargs["__no_parallel"]
        ):
            ret = list(starmap(partial_func, map_input))
        else:
            with self.ctx.Pool(self.n_jobs) as p:
                ret = p.starmap(partial_func, map_input)
        return ret

    def async_apply(
            self,
            function: Callable,
            n_iterations: int,
            **kwargs) -> Iterable:
        """
        Applies the function n_iterations number of times and returns the result
        of the n_iterations in an unknown order.


        Parameters
        ----------
        function
            Function to apply
        n_iterations
            Number of applications of function

        Returns
        -------
        Iterable
            Function applied n_iterations number of times
        """
        partial_func = partial(function, **kwargs)
        if self.n_jobs == 1 or ("__no_parallel" in kwargs):
            ret = [partial_func() for _ in range(n_iterations)]
        else:
            with self.ctx.Pool(self.n_jobs) as p:
                promise = [p.apply_async(partial_func)
                           for _ in range(n_iterations)]
                ret = [res.get() for res in promise]
        return ret

    def apply(
            self,
            function: Callable,
            n_iterations: int,
            **kwargs) -> Iterable:
        """
        Applies the function n_iterations number of times and returns the result
        of the n_iterations where element i corresponds to the i'th return value
        of function.

        Parameters
        ----------
        function
            Function to apply
        n_iterations
            Number  of applications of function

        Returns
        -------
        Iterable
            Function applied n_iterations number of times
        """
        partial_func = partial(function, **kwargs)
        if self.n_jobs == 1 or ("__no_parallel" in kwargs):
            ret = [partial_func() for _ in range(n_iterations)]
        else:
            with self.ctx.Pool(self.n_jobs) as p:
                ret = [p.apply(partial_func) for _ in range(n_iterations)]
        return ret
