from copy import deepcopy
import numpy as np


class Bagging:
    """
    Bagging / bootstrap aggregation for binary
    classification.
    """

    def __init__(self, base_clf, M):
        """
        Initialize a bootstrap aggregation model.

        Parameters
        ----------
        base_clf : object
            base classifier to use
        M : int
            number of bootstrap samples
        """
        self.base_clf = base_clf
        self.M = M
        self.clfs = []

    def fit(self, X, y):
        """
        Fit models to data.

        Parameters
        ----------
        X : (N, p) np.ndarray
            observations
        y : (N,) np.ndarray
            targets

        Returns
        -------
        clfs : Bagging
            fitted classifiers
        """
        self.clfs = []
        for _ in range(self.M):
            # bootstrap data
            idx = np.random.choice(range(len(X)), size=len(X), replace=True)
            X_m = X[idx]
            y_m = y[idx]
            clf = deepcopy(self.base_clf)
            clf.fit(X_m, y_m)
            self.clfs.append(clf)
        return self

    def predict(self, X):
        """
        Predict on input data. Note
        that class labels are {-1, 1}.

        Parameters
        ----------
        X : (N, p) np.ndarray
            observations

        Returns
        -------
        y_hat : (N,) np.ndarray
            predictions
        """
        y_hat = sum([clf_m.predict(X) for clf_m in self.clfs])
        return np.sign(y_hat)


class AdaBoost:
    """
    AdaBoost.M1 boosting algorithm for binary
    classification.
    """

    def __init__(self, base_clf, M):
        """
        Initialize AdaBoost.

        Parameters
        ----------
        base_clf : object
            base classifier to use
        M : int
            number of classifiers
        """
        self.base_clf = base_clf
        self.M = M
        self.clfs = []
        self.alphas = []

    def fit(self, X, y):
        """
        Fit AdaBoost to data.

        Parameters
        ----------
        X : (N, p) np.ndarray
            observations
        y : (N,) np.ndarray
            class labels

        Returns
        -------
        clf : AdaBoost
            fitted AdaBoost
        """
        w = np.ones(len(X)) / len(X)

        self.clfs = []
        self.alphas = []
        for _ in range(self.M):
            clf_m = deepcopy(self.base_clf)
            clf_m.fit(X, y, w=w)
            y_hat = clf_m.predict(X)
            err_m = sum(w[y_hat != y]) / sum(w)
            alpha_m = np.log(1.0 - err_m) - np.log(err_m + 1e-12)
            w = w * np.exp(alpha_m * (y_hat != y))
            self.clfs.append(clf_m)
            self.alphas.append(alpha_m)
        return self

    def predict(self, X):
        """
        Predict the class labels for input observations. Note
        that class labels are {-1, 1}.

        Parameters
        ----------
        X : (N, p) np.ndarray
            observations

        Returns
        -------
        y_hat : (N,) np.ndarray
            class predictions
        """
        y_hat = sum(
            [
                alpha_m * clf_m.predict(X)
                for alpha_m, clf_m in zip(self.alphas, self.clfs)
            ]
        )
        return np.sign(y_hat)
