# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
from numpy import float64 as DOUBLE
from .predict import Predict
from .criteria import Criteria
from .criteria.criteria import Entropy, Squared_error, Partial_quadratic
from .decision_tree.splitter import Splitter
from .leaf_builder import LeafBuilder

from .predict.predict cimport (PredictClassification, PredictRegression,
                               PredictLocalPolynomial, PredictQuantile)
from .leaf_builder.leaf_builder cimport (LeafBuilderClassification,
                                         LeafBuilderRegression,
                                         LeafBuilderPartialQuadratic)

import numpy as np


class BaseModel:

    def _check_max_features(
        self, max_features: int | str | float | None
    ) -> int | str | float | None:

        if max_features is None:
            return max_features
        elif isinstance(max_features, int):
            if max_features < 1:
                raise ValueError("max_features can not be less than 1")
            else:
                return max_features
        elif isinstance(max_features, float):
            return max_features
        elif isinstance(max_features, str):
            if max_features == "sqrt":
                return max_features
            elif max_features == "log2":
                return max_features
            else:
                raise ValueError("The only string options available for max_features are \"sqrt\", \"log2\"")
        else:
            raise ValueError("max_features can only be int, float, or in {\"sqrt\", \"log2\"}")

    def _check_sample_weight(self, sample_weight: ArrayLike | None) -> np.ndarray:
        if sample_weight is None:
            return np.ones(self.X_n_rows, dtype=DOUBLE)
        sample_weight = np.array(sample_weight, dtype=DOUBLE)
        if sample_weight.shape[0] != self.X_n_rows:
            raise ValueError("sample_weight should have as many elements as X and Y")
        if sample_weight.ndim > 1:
            raise ValueError("sample_weight has more than one dimension")
        return sample_weight

    def _check_sample_indices(self,
                              sample_indices: ArrayLike | None
                              ) -> np.ndarray:

        if sample_indices is None:
            return np.arange(0, self.X_n_rows, dtype=np.int32)
        else:
            sample_indices = np.array(sample_indices, dtype=np.int32)
        if sample_indices.ndim > 1:
            raise ValueError("sample_weight has more than one dimension")
        return np.array(sample_indices, dtype=np.int32)

    # Check whether dimension of X matches self.n_features
    def _check_dimensions(self, X: np.ndarray) -> None:
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Number of features should be {self.n_features}, got {X.shape[1]}"
            )

    def _check_input(self,
                     X: ArrayLike,
                     Y: ArrayLike | None = None
                     ) -> tuple[np.ndarray, np.ndarray|None]:

        Y_check = (Y is not None)
        # Make sure input arrays are c contigous
        X = np.ascontiguousarray(X, dtype=DOUBLE)

        # Check that X is two dimensional
        if X.ndim > 2:
            raise ValueError("X should not be more than 2 dimensions")
        elif X.ndim == 1:
            X = np.expand_dims(X, axis=1)
        elif X.ndim < 1:
            raise ValueError("X has less than 1 dimension")

        # If Y is not None perform checks for Y
        if Y_check:
            Y = np.ascontiguousarray(Y, dtype=DOUBLE)
            # Check if X and Y has same number of rows
            if X.shape[0] != Y.shape[0]:
                raise ValueError("X and Y should have the same number of rows")

            # Check if Y has dimensions (n, 1) or (n,)
            if 2 < Y.ndim:
                raise ValueError("Y should not have more than 2 dimensions")
            elif Y.ndim == 1:
                Y = np.expand_dims(Y, axis=1)
            elif Y.ndim < 1:
                raise ValueError("Y has less than 1 dimension")
        return X, Y

    def _check_tree_type(
        self,
        tree_type: str | None,
        criteria: type[Criteria] | None,
        splitter: type[Splitter] | None,
        leaf_builder: type[LeafBuilder] | None,
        predict: type[Predict] | None,
    ) -> None:
        tree_types = ["Classification", "Regression", "Gradient", "Quantile"]
        if tree_type in tree_types:
            if tree_type == "Classification":
                if predict:
                    self.predict_class = predict
                else:
                    self.predict_class = PredictClassification
                if criteria:
                    self.criteria_class = criteria
                else:
                    self.criteria_class = Entropy
                if leaf_builder:
                    self.leaf_builder_class = leaf_builder
                else:
                    self.leaf_builder_class = LeafBuilderClassification
            elif tree_type == "Regression":
                if predict:
                    self.predict_class = predict
                else:
                    self.predict_class = PredictRegression
                if criteria:
                    self.criteria_class = criteria
                else:
                    self.criteria_class = Squared_error
                if leaf_builder:
                    self.leaf_builder_class = leaf_builder
                else:
                    self.leaf_builder_class = LeafBuilderRegression
            elif tree_type == "Quantile":
                if predict:
                    self.predict_class = predict
                else:
                    self.predict_class = PredictQuantile
                if criteria:
                    self.criteria_class = criteria
                else:
                    self.criteria_class = Squared_error
                if leaf_builder:
                    self.leaf_builder_class = leaf_builder
                else:
                    self.leaf_builder_class = LeafBuilderRegression
            elif tree_type == "Gradient":
                if predict:
                    self.predict_class = predict
                else:
                    self.predict_class = PredictLocalPolynomial
                if criteria:
                    self.criteria_class = criteria
                else:
                    self.criteria_class = Partial_quadratic
                if leaf_builder:
                    self.leaf_builder_class = leaf_builder
                else:
                    self.leaf_builder_class = LeafBuilderPartialQuadratic

        else:
            if (not criteria) or (not predict) or (not leaf_builder):
                raise ValueError(
                    "tree_type was not a default tree_type, so criteria, predict and leaf_builder must be supplied"
                )
            self.criteria_class = criteria
            self.predict_class = predict
            self.leaf_builder_class = leaf_builder

        if splitter:
            self.splitter_class = splitter
        else:
            self.splitter_class = Splitter
