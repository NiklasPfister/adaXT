from adaXT.decision_tree.leafbuilder import LinearRegressionLeafBuilder
from adaXT.decision_tree.predict import PredictLinearRegression
from adaXT.decision_tree import DecisionTree
from adaXT.criteria import Linear_regression
import numpy as np

X = np.random.uniform(0, 100, (1000, 5))
Y = np.random.uniform(0, 100, 1000)

tree = DecisionTree(
    "LinearRegression",
    criteria=Linear_regression,
    predict=PredictLinearRegression,
    leaf_builder=LinearRegressionLeafBuilder,
)
tree.fit(X, Y)

print(tree.predict(X))
