from functools import partial
import multiprocessing
from multiprocessing import cpu_count
from itertools import starmap
from typing import Any, Callable, Iterable


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
        n_jobs : int, default=1
            The number of processes used to fit, and predict for the forest, -1 uses all available proccesors
        random_state: int
            Used for deterministic seeding of the tree
        """
        self.ctx = multiprocessing.get_context("fork")
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()

    def async_map(self, function: Callable, map_input: Any, **kwargs):
        partial_func = partial(function, **kwargs)
        if self.n_jobs == 1:
            ret = list(map(partial_func, map_input))
        else:
            with self.ctx.Pool(self.n_jobs) as p:
                promise = p.map_async(partial_func, map_input)
                ret = promise.get()
        return ret

    def map(self, function: Callable, map_input: Any, **kwargs):
        partial_func = partial(function, **kwargs)
        if self.n_jobs == 1:
            ret = list(map(partial_func, map_input))
        else:
            with self.ctx.Pool(self.n_jobs) as p:
                ret = p.map(partial_func, map_input)
        return ret

    def async_starmap(self, function: Callable, map_input: Iterable, **kwargs):
        partial_func = partial(function, **kwargs)
        if self.n_jobs == 1:
            ret = list(starmap(partial_func, map_input))
        else:
            with self.ctx.Pool(self.n_jobs) as p:
                promise = p.starmap_async(partial_func, map_input)
                ret = promise.get()
        return ret

    def starmap(self, function: Callable, map_input: Iterable, **kwargs):
        partial_func = partial(function, **kwargs)
        if self.n_jobs == 1:
            ret = list(starmap(partial_func, map_input))
        else:
            with self.ctx.Pool(self.n_jobs) as p:
                ret = p.starmap(partial_func, map_input)
        return ret

    def async_apply(self, function: Callable, n_iterations: int, **kwargs):
        partial_func = partial(function, **kwargs)
        if self.n_jobs == 1:
            ret = [partial_func() for _ in range(n_iterations)]
        else:
            with self.ctx.Pool(self.n_jobs) as p:
                promise = [p.apply_async(partial_func) for _ in range(n_iterations)]
                ret = [res.get() for res in promise]
        return ret

    def apply(self, function: Callable, n_iterations: int, **kwargs):
        partial_func = partial(function, **kwargs)
        if self.n_jobs == 1:
            ret = [partial_func() for _ in range(n_iterations)]
        else:
            with self.ctx.Pool(self.n_jobs) as p:
                ret = [p.apply(partial_func) for _ in range(n_iterations)]
        return ret
