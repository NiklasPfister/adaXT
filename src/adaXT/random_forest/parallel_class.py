from functools import partial
import multiprocessing
from multiprocessing import cpu_count
from multiprocessing.managers import BaseManager
from numbers import Integral
from types import FunctionType
from typing import Any

import numpy as np


class ParallelModel:
    """
    Class used to parallelize
    """

    def __init__(
        self,
        n_jobs: int = -1,
        random_state: int | None = None,
    ):
        """
        Parameters
        ----------
        n_jobs : int, default=1
            The number of processes used to fit, and predict for the forest, -1 uses all available proccesors
        random_state: int
            Used for deterministic seeding of the tree
        """
        self.ctx = multiprocessing.get_context("spawn")
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        BaseManager.register("RandomState", np.random.RandomState)
        self.manager = BaseManager()
        self.manager.start()
        self.random_state = self.__get_random_state(random_state)

    def __get_random_state(self, random_state):
        if isinstance(random_state, Integral) or (random_state is None):
            return self.manager.RandomState(random_state)
        else:
            raise ValueError("Random state either has to be Integral or None")

    def async_map_multiple(self, functions: list[FunctionType], constant_inputs:
                           list[dict], map_inputs: list[Any]):
        partial_funcs = [partial(func, **const)
                         for func, const in zip(functions, constant_inputs)]
        ret = []
        if self.n_jobs == 1:
            for f, i in zip(partial_funcs, map_inputs):
                ret.append(list(map(f, i)))
        else:
            with self.ctx.Pool(self.n_jobs) as p:
                for f, i in zip(partial_funcs, map_inputs):
                    promise = p.map_async(f, i)
                    ret.append(promise.get())
        return ret

    def async_map_single(self, function: FunctionType, constant_input:
                         dict, map_input: Any):
        partial_func = partial(function, **constant_input)
        if self.n_jobs == 1:
            ret = (list(map(function, map_input)))
        else:
            with self.ctx.Pool(self.n_jobs) as p:
                promise = p.map_async(partial_func, map_input)
                ret = promise.get()
        return ret
