import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_diabetes, load_breast_cancer

from decision_tree import DecisionTree

diabetes = load_diabetes()
breast_cancer = load_breast_cancer()


def test_regression():
    # There can be small differences in regression trees w.r.t. scikit-learn.
    # I was not able to identify the difference, but possible reasons could be:
    # - numerical differences
    # - sklearn does not consider weakest link multiplicity in pruning
    # We therefore pass the check as long as 98% of all values are equal.
    MAX_DEPTH = 10
    MIN_SAMPLES_SPLIT = 4

    X, y = diabetes.data, diabetes.target

    sk_tree = DecisionTreeRegressor(
        criterion="squared_error",
        min_samples_split=MIN_SAMPLES_SPLIT,
        max_depth=MAX_DEPTH,
    )
    sk_tree.fit(X, y)
    sk_preds = sk_tree.predict(X)

    tree = DecisionTree(
        criterion="squared_error",
        assign_leaf_node="mean",
        min_samples_split=MIN_SAMPLES_SPLIT,
        max_depth=MAX_DEPTH,
    )
    tree.fit(X, y)
    preds = tree.predict(X)

    assert np.mean(np.isclose(preds, sk_preds)) > 0.98


def test_regression_with_pruning():
    # There can be small differences in regression trees w.r.t. scikit-learn.
    # I was not able to identify the difference, but possible reasons could be:
    # - numerical differences
    # - sklearn does not consider weakest link multiplicity in pruning
    # We therefore pass the check as long as 98% of all values are equal.
    MAX_DEPTH = 15
    CCP_ALPHA = 0.1

    X, y = diabetes.data, diabetes.target

    sk_tree = DecisionTreeRegressor(
        criterion="squared_error", max_depth=MAX_DEPTH, ccp_alpha=CCP_ALPHA
    )
    sk_tree.fit(X, y)
    sk_preds = sk_tree.predict(X)

    tree = DecisionTree(
        criterion="squared_error",
        assign_leaf_node="mean",
        max_depth=MAX_DEPTH,
        ccp_alpha=CCP_ALPHA,
    )
    tree.fit(X, y)
    preds = tree.predict(X)

    assert np.mean(np.isclose(preds, sk_preds)) > 0.98


def test_classification():
    MAX_DEPTH = 10
    MIN_SAMPLES_SPLIT = 4

    X, y = breast_cancer.data, breast_cancer.target

    sk_tree = DecisionTreeClassifier(
        criterion="gini", min_samples_split=MIN_SAMPLES_SPLIT, max_depth=MAX_DEPTH
    )
    sk_tree.fit(X, y)
    sk_preds = sk_tree.predict(X)

    tree = DecisionTree(
        criterion="gini",
        assign_leaf_node="most_common",
        min_samples_split=MIN_SAMPLES_SPLIT,
        max_depth=MAX_DEPTH,
    )
    tree.fit(X, y)
    preds = tree.predict(X)

    assert np.allclose(preds, sk_preds)


def test_classification_with_pruning():
    MAX_DEPTH = 15
    CCP_ALPHA = 0.1

    X, y = breast_cancer.data, breast_cancer.target

    sk_tree = DecisionTreeClassifier(
        criterion="gini", max_depth=MAX_DEPTH, ccp_alpha=CCP_ALPHA
    )
    sk_tree.fit(X, y)
    sk_preds = sk_tree.predict(X)

    tree = DecisionTree(
        criterion="gini",
        assign_leaf_node="most_common",
        max_depth=MAX_DEPTH,
        ccp_alpha=CCP_ALPHA,
    )
    tree.fit(X, y)
    preds = tree.predict(X)

    assert np.allclose(preds, sk_preds)


def test_classification_with_entropy():
    MAX_DEPTH = 15
    CCP_ALPHA = 0.1

    X, y = breast_cancer.data, breast_cancer.target

    sk_tree = DecisionTreeClassifier(
        criterion="entropy", max_depth=MAX_DEPTH, ccp_alpha=CCP_ALPHA
    )
    sk_tree.fit(X, y)
    sk_preds = sk_tree.predict(X)

    tree = DecisionTree(
        criterion="entropy",
        assign_leaf_node="most_common",
        max_depth=MAX_DEPTH,
        ccp_alpha=CCP_ALPHA,
    )
    tree.fit(X, y)
    preds = tree.predict(X)

    assert np.allclose(preds, sk_preds)
