from .predictor import Predictor
from .criteria import Criteria
from .decision_tree.splitter import Splitter
from .leaf_builder import LeafBuilder
from typing import Type

import numpy as np
from numpy.typing import ArrayLike

class BaseModel:
    predictor: Type[Predictor] | None
    leaf_builder: Type[LeafBuilder] | None
    criteria: Type[Criteria] | None
    splitter: Type[Splitter] | None

    def _check_max_features(
        self, max_features: int | str | float | None
    ) -> int | str | float | None:
        pass

    def _check_sample_weight(self, sample_weight: ArrayLike | None) -> np.ndarray:
        pass

    def _check_sample_indices(self, sample_indices: ArrayLike | None) -> np.ndarray:
        pass

    # Check whether dimension of X matches self.n_features
    def _check_dimensions(self, X: np.ndarray) -> None:
        pass

    def _check_input(
        self, X: ArrayLike, Y: ArrayLike | None = None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        pass

    def _check_tree_type(
        self,
        tree_type: str | None,
        criteria: type[Criteria] | None,
        splitter: type[Splitter] | None,
        leaf_builder: type[LeafBuilder] | None,
        predictor: type[Predictor] | None,
    ):
        pass

    def predict(self, X: ArrayLike, **kwargs) -> np.ndarray:
        pass

    def similarity(self, X0: np.ndarray, X1: np.ndarray) -> np.ndarray:
        pass

    def predict_weights(self, X: np.ndarray | None, scale: bool = True) -> np.ndarray:
        pass
