from functools import partial
import multiprocessing
from multiprocessing import cpu_count
from multiprocessing.managers import BaseManager
from itertools import starmap
from numbers import Integral
from typing import Any, Callable, Iterable

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

    def async_map(self, function: Callable, map_input: Any, **kwargs):
        partial_func = partial(function, **kwargs)
        print(partial_func, map_input)
        if self.n_jobs == 1:
            ret = list(map(partial_func, map_input))
        else:
            with self.ctx.Pool(self.n_jobs) as p:
                promise = p.map_async(partial_func, map_input)
                ret = promise.get()
        return ret

    def async_starmap(self, function: Callable, map_input: Iterable,
                      **kwargs):
        partial_func = partial(function, **kwargs)
        if self.n_jobs == 1:
            ret = list(starmap(partial_func, map_input))
        else:
            with self.ctx.Pool(self.n_jobs) as p:
                promise = p.starmap_async(partial_func, map_input)
                ret = promise.get()
        return ret

    def async_apply(self, function: Callable, n_iterations: int, **kwargs):
        partial_func = partial(function, **kwargs)
        if self.n_jobs == 1:
            ret = [partial_func() for _ in range(n_iterations)]
        else:
            with self.ctx.Pool(self.n_jobs) as p:
                promise = [p.apply_async(partial_func)
                           for _ in range(n_iterations)]
                ret = [res.get() for res in promise]
        return ret
