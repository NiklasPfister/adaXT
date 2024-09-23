from .predict import Predict
from .criteria import Criteria
from .decision_tree.splitter import Splitter
from .leaf_builder import LeafBuilder
from typing import Type

import numpy as np
from numpy.typing import ArrayLike

class BaseModel:
    predict_class: Type[Predict]
    leaf_builder_class: Type[Criteria]
    criteria_class: Type[LeafBuilder]

    def _check_sample_weight(self, sample_weight: ArrayLike | None) -> np.ndarray:
        pass

    def _check_sample_indices(self, sample_indices: ArrayLike | None) -> np.ndarray:
        pass

    # Check whether dimension of X matches self.n_features
    def _check_dimensions(self, X: np.ndarray) -> None:
        pass

    def _check_input(
        self, X: ArrayLike, Y: ArrayLike | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        pass

    def _check_tree_type(
        self,
        tree_type: str | None,
        criteria: type[Criteria] | None,
        splitter: type[Splitter] | None,
        leaf_builder: type[LeafBuilder] | None,
        predict: type[Predict] | None,
    ):
        pass

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        sample_indices: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ):
        pass

    def predict(self, X: np.ndarray, Y: np.ndarray, **kwargs):
        pass
