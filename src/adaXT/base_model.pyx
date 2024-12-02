# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
from numpy import float64 as DOUBLE
from .predictor import Predictor
from .criteria import Criteria
from .criteria.criteria import Entropy, Squared_error, Partial_quadratic
from .decision_tree.splitter import Splitter
from .leaf_builder import LeafBuilder

from .predictor.predictor cimport (PredictorClassification, PredictorRegression,
                               PredictorLocalPolynomial, PredictorQuantile)
from .leaf_builder.leaf_builder cimport (LeafBuilderClassification,
                                         LeafBuilderRegression,
                                         LeafBuilderPartialQuadratic)

import numpy as np
from collections import defaultdict
from numpy.typing import ArrayLike

import inspect


class BaseModel():

    def _check_max_features(
        self, max_features: int | str | float | None, tot_features: int
    ) -> int:

        if max_features is None:
            return -1
        elif isinstance(max_features, int):
            if max_features < 1:
                raise ValueError("max_features can not be less than 1")
            else:
                return min(max_features, tot_features)
        elif isinstance(max_features, float):
            return min(tot_features, int(max_features * tot_features))
        elif isinstance(max_features, str):
            if max_features == "sqrt":
                return int(np.sqrt(tot_features))
            elif max_features == "log2":
                return int(np.log2(tot_features))
            else:
                raise ValueError("The only string options available for max_features are \"sqrt\", \"log2\"")
        else:
            raise ValueError("max_features can only be int, float, or in {\"sqrt\", \"log2\"}")

    def _check_sample_weight(self, sample_weight: ArrayLike | None, X_n_rows : int |None = None) -> np.ndarray:
        if X_n_rows is None:
            X_n_rows = self.X_n_rows
        if sample_weight is None:
            return np.ones(X_n_rows, dtype=DOUBLE)
        sample_weight = np.array(sample_weight, dtype=DOUBLE)
        if sample_weight.shape[0] != X_n_rows:
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
                     X: ArrayLike | None = None,
                     Y: ArrayLike | None = None
                     ) -> tuple[np.ndarray|None, np.ndarray|None]:

        if (X is None) and (Y is None):
            raise ValueError(
                    "X and Y are both None when checking input"
                    )
        if X is not None:
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
        if Y is not None:
            Y = np.ascontiguousarray(Y, dtype=DOUBLE)

            # Check if Y has dimensions (n, 1) or (n,)
            if 2 < Y.ndim:
                raise ValueError("Y should not have more than 2 dimensions")
            elif Y.ndim == 1:
                Y = np.expand_dims(Y, axis=1)
            elif Y.ndim < 1:
                raise ValueError("Y has less than 1 dimension")

        if (Y is not None) and (X is not None):
            # Check if X and Y has same number of rows
            if X.shape[0] != Y.shape[0]:
                raise ValueError("X and Y should have the same number of rows")
        return X, Y

    def _check_tree_type(
        self,
        tree_type: str | None,
        criteria: type[Criteria] | None,
        splitter: type[Splitter] | None,
        leaf_builder: type[LeafBuilder] | None,
        predictor: type[Predictor] | None,
    ) -> None:
        # tree_types. To add a new one add an entry in the following dictionary,
        # where the key is the name, and the value is a list of a criteria,
        # predict and leaf_builder class in that order.
        tree_types = {
                "Classification": [Entropy, PredictorClassification,
                                   LeafBuilderClassification],
                "Regression": [Squared_error, PredictorRegression, LeafBuilderRegression],
                "Gradient": [Partial_quadratic, PredictorLocalPolynomial, LeafBuilderPartialQuadratic],
                "Quantile": [Squared_error, PredictorQuantile, LeafBuilderRegression]
            }
        if tree_type in tree_types.keys():
            # Set the defaults
            self.criteria, self.predictor, self.leaf_builder = \
                tree_types[tree_type]

            # Update any that are specifically given
            if criteria is not None:
                self.criteria = criteria
            if splitter is not None:
                self.splitter = splitter
            if leaf_builder is not None:
                self.leaf_builder = leaf_builder
            if predictor is not None:
                self.predictor = predictor
        else:
            if (criteria is None) or (predictor is None) or (leaf_builder is
                                                             None):
                print(criteria, predictor, leaf_builder)
                raise ValueError(
                    "tree_type was not a default tree_type, so criteria, predictor and leaf_builder must be supplied"
                )
            self.criteria = criteria
            self.predictor = predictor
            self.leaf_builder = leaf_builder

        if splitter is None:
            self.splitter = Splitter
        else:
            self.splitter = splitter

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "scikit-learn estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def score(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike|None = None):
        X, Y = self._check_input(X, y)
        _, Y_pred = self._check_input(None, self.predict(X))
        _, Y_true = self._check_input(None, Y)
        sample_weight = self._check_sample_weight(sample_weight, X.shape[0])
        return -self.criteria.loss(Y_pred, Y_true, sample_weight)
