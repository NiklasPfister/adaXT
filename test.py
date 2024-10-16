from adaXT.random_forest import RandomForest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn import datasets
import copy
import inspect


def score(clf, X, Y):
    return clf.score(X, Y)


iris = datasets.load_iris()
X = iris.data
Y = iris.target

rf = RandomForest("Classification")
parameters = {"max_features": range(0, 4), "max_depth": range(0, 4)}

params = rf.get_params()


def clone(estimator, *, safe=True):
    if hasattr(estimator, "__sklearn_clone__") and not inspect.isclass(estimator):
        return estimator.__sklearn_clone__()
    return _clone_parametrized(estimator, safe=safe)


def _clone_parametrized(estimator, *, safe=True):
    estimator_type = type(estimator)
    if estimator_type is dict:
        return {k: clone(v, safe=safe) for k, v in estimator.items()}
    elif estimator_type in (list, tuple, set, frozenset):
        return estimator_type([clone(e, safe=safe) for e in estimator])
    elif not hasattr(estimator, "get_params") or isinstance(estimator, type):
        if not safe:
            return copy.deepcopy(estimator)
        else:
            if isinstance(estimator, type):
                raise TypeError(
                    "Cannot clone object. "
                    + "You should provide an instance of "
                    + "scikit-learn estimator instead of a class."
                )
            else:
                raise TypeError(
                    "Cannot clone object '%s' (type %s): "
                    "it does not seem to be a scikit-learn "
                    "estimator as it does not implement a "
                    "'get_params' method." % (repr(estimator), type(estimator))
                )

    klass = estimator.__class__
    new_object_params = estimator.get_params(deep=False)
    for name, param in new_object_params.items():
        new_object_params[name] = clone(param, safe=False)

    print("Id in new_object_params: ", id(new_object_params["impurity_tol"]))
    new_object = klass(**new_object_params)
    try:
        new_object._metadata_request = copy.deepcopy(estimator._metadata_request)
    except AttributeError:
        pass

    print("Id on new_object", id(new_object.impurity_tol))
    params_set = new_object.get_params(deep=False)

    # quick sanity check of the parameters of the clone
    for name in new_object_params:
        param1 = new_object_params[name]
        param2 = params_set[name]
        if param1 is not param2:
            print(id(param1), id(param2))
            raise RuntimeError(
                "Cannot clone object %s, as the constructor "
                "either does not set or modifies parameter %s" % (estimator, name)
            )

    # _sklearn_output_config is used by `set_output` to configure the output
    # container of an estimator.
    if hasattr(estimator, "_sklearn_output_config"):
        new_object._sklearn_output_config = copy.deepcopy(
            estimator._sklearn_output_config
        )
    return new_object


print("Id on forest", id(rf.impurity_tol))
base_estimator = _clone_parametrized(rf)


# clf = GridSearchCV(rf, parameters, scoring=make_scorer(score, greater_is_better=False))
# clf.fit(X, Y)
# print(sorted(clf.cv_results_.keys()))
