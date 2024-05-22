# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

from .predict import Predict
from .criteria import Criteria
from .criteria.criteria import Entropy, Squared_error, Linear_regression
from .decision_tree.splitter import Splitter
from .leaf_builder import LeafBuilder

from .predict.predict cimport PredictClassification, PredictRegression, PredictLinearRegression, PredictQuantile
from .leaf_builder.leaf_builder cimport LeafBuilderClassification, LeafBuilderRegression, LeafBuilderLinearRegression

import numpy as np


class BaseModel:
    def check_tree_type(
        self,
        tree_type: str | None,
        criteria: type[Criteria] | None,
        splitter: type[Splitter] | None,
        leaf_builder: type[LeafBuilder] | None,
        predict: type[Predict] | None,
    ):
        tree_types = ["Classification", "Regression", "LinearRegression", "Quantile"]
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
            elif tree_type == "LinearRegression":
                if predict:
                    self.predict_class = predict
                else:
                    self.predict_class = PredictLinearRegression
                if criteria:
                    self.criteria_class = criteria
                else:
                    self.criteria_class = Linear_regression
                if leaf_builder:
                    self.leaf_builder_class = leaf_builder
                else:
                    self.leaf_builder_class = LeafBuilderLinearRegression

        else:
            if (not criteria) or (not predict) or (not leaf_builder):
                raise ValueError(
                    "tree_type was not a default tree_type, so criteria, predict and leaf_builder must be supplied"
                )
            self.criteria_class = criteria
            self.predict_class = predict
            self.leaf_builder_class = leaf_builder

        if splitter:
            self.splitter = splitter
        else:
            self.splitter = Splitter

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

    def predict_proba(self, X: np.ndarray):
        pass
