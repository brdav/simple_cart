import numpy as np


class TreeNode:
    """
    Class to represent a single node in a decision tree.
    """

    def __init__(self, left, right, col_idx, threshold, value, impurity, samples):
        """
        Single node in a tree. In this representation 'True' values for
        a decision take us to the left.

        Parameters
        ----------
        left : TreeNode
            left child node
        right : TreeNode
            right child node
        col_idx : int
            index of feature to split
        threshold : float
            threshold value for splitting
        value : float
            target value of the node
        impurity : float
            impurity of the node
        samples : float
            effective number of observations
        """
        self.left = left
        self.right = right
        self.col_idx = col_idx
        self.threshold = threshold
        self.value = value
        self.impurity = impurity
        self.samples = samples

    def decide(self, x_n):
        """
        Traverse node based on the decision function.

        Parameters
        ----------
        x_n : (p,) np.ndarray
            observation

        Returns
        -------
        ret : float or int
            decision value for current branch
        """
        if self.left is None and self.right is None:
            return self.value
        elif x_n[self.col_idx] < self.threshold:
            return self.left.decide(x_n)
        else:
            return self.right.decide(x_n)


class DecisionTree:
    """
    CART-based decision tree.
    """

    def __init__(
        self,
        criterion="gini",
        assign_leaf_node="most_common",
        max_depth=float("inf"),
        min_samples_split=2,
        ccp_alpha=0.0,
        max_features="none",
    ):
        """
        Initialize the decision tree.

        Parameters
        ----------
        criterion : string
            name of the splitting criterion to use
        assign_leaf_node : str
            how to assign leaf node values
        max_depth : int
            maximum depth of the tree
        min_samples_split : int
            minimum number of samples required to split
        ccp_alpha : float
            regularization parameter for pruning
        max_features : str
            "none" or "sqrt", maximum number of features to
            consider at each split
        """
        self.root = None
        self.criterion = getattr(self, criterion)
        self.assign_leaf_node = assign_leaf_node
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.ccp_alpha = ccp_alpha
        self.max_features = max_features

    def fit(self, X, y, w=None):
        """
        Fit the decision tree to data. First grow the tree
        and optionally prune it.

        Parameters
        ----------
        X : (N, p) np.ndarray
            observations
        y : (N,) np.ndarray
            class labels
        w : (N,) np.ndarray, optional
            weigh samples in criterion

        Returns
        -------
        model : DecisionTree
            fitted decision tree
        """
        if w is None:
            w = np.ones(X.shape[0])
        self.root = self._grow_tree(X, y, w)
        if self.ccp_alpha > 0:
            self._minimal_cost_complexity_pruning(sum(w))
        return self

    def _grow_tree(self, X, y, w, depth=0):
        """
        Depth-first strategy to build tree greedily.

        Parameters
        ----------
        X : (N, p) np.ndarray
            observations
        y : (N,) np.ndarray
            class labels
        w : (N,) np.ndarray
            weigh samples in criterion
        depth : int
            current tree depth

        Returns
        -------
        node : TreeNode
            node to add to tree
        """
        best_gain = 0
        best_column_index = None
        best_column_threshold = None

        impurity_base = self.criterion(y, w)

        if (
            (len(y) < self.min_samples_split)
            or (depth == self.max_depth)
            or np.all(y[0] == y[:])
        ):
            return TreeNode(
                None,
                None,
                None,
                None,
                self.node_value(y, w),
                impurity_base,
                sum(w),
            )
        else:  # build tree recursively
            if self.max_features == "sqrt":
                select_cols = np.random.choice(
                    np.arange(X.shape[1]), int(np.sqrt(X.shape[1])), replace=False
                )
            else:
                select_cols = np.arange(X.shape[1])
            for column_i in select_cols:
                split_vals = (
                    np.unique(X[:, column_i])[:-1] + np.unique(X[:, column_i])[1:]
                ) / 2
                for split_val in split_vals:
                    y_left = y[X[:, column_i] < split_val]
                    w_left = w[X[:, column_i] < split_val]
                    y_right = y[X[:, column_i] >= split_val]
                    w_right = w[X[:, column_i] >= split_val]
                    # getting the left and right losses
                    impurity_left = self.criterion(y_left, w_left)
                    impurity_right = self.criterion(y_right, w_right)
                    # calculating the weights for each of the nodes
                    impurity_split = (
                        sum(w_left) * impurity_left + sum(w_right) * impurity_right
                    ) / sum(w)
                    column_i_gain = impurity_base - impurity_split
                    if column_i_gain > best_gain:
                        best_gain = column_i_gain
                        best_column_index = column_i
                        best_column_threshold = split_val

            if best_gain <= 0:
                # stop recursion when there is no more gain
                return TreeNode(
                    None, None, None, None, self.node_value(y, w), impurity_base, sum(w)
                )

            # divide dataset and continue recursion
            X_left = X[X[:, best_column_index] < best_column_threshold]
            y_left = y[X[:, best_column_index] < best_column_threshold]
            w_left = w[X[:, best_column_index] < best_column_threshold]
            X_right = X[X[:, best_column_index] >= best_column_threshold]
            y_right = y[X[:, best_column_index] >= best_column_threshold]
            w_right = w[X[:, best_column_index] >= best_column_threshold]

            left_tree = self._grow_tree(X_left, y_left, w_left, depth=depth + 1)
            right_tree = self._grow_tree(X_right, y_right, w_right, depth=depth + 1)

            return TreeNode(
                left_tree,
                right_tree,
                best_column_index,
                best_column_threshold,
                self.node_value(y, w),
                impurity_base,
                sum(w),
            )

    def _minimal_cost_complexity_pruning(self, N):
        """
        Minimal cost-complexity pruning, implemented
        through iterative weakest-link pruning. For
        a description of the algorithm see

        Breiman, L. "Classification and regression trees." Routledge, 2017

        Parameters
        ----------
        node : TreeNode
            current tree node
        N : int
            total number of observations

        Returns
        -------
        impurity : float
            the impurity of the current subtree
        T : int
            the number of leaf nodes of current subtree
        """
        while True:
            weakest_links, alpha_eff, _, _ = self._find_weakest_links(
                self.root, N, [], float("inf")
            )

            if alpha_eff <= self.ccp_alpha:
                for node in weakest_links:
                    # prune the weakest link
                    node.left = None
                    node.right = None
                    node.col_idx = None
                    node.threshold = None

            else:
                break

    def _find_weakest_links(self, node, N, weakest_links, alpha_eff):
        """
        Find weakest links recursively (depth-first search).

        Parameters
        ----------
        node : TreeNode
            current tree node
        N : int
            total number of observations
        weakest_links : list
            weakest links before traversing this node
        alpha_eff : float
            alpha_eff of weakest links before traversing this node

        Returns
        -------
        weakest_links : list
            weakest links after traversing this node
        alpha_eff : float
            alpha_eff of weakest links after traversing this node
        impurity : float
            the impurity of this branch
        T : int
            the number of leaf nodes of this branch
        """
        if not (node.left or node.right):
            return weakest_links, alpha_eff, node.impurity * node.samples / N, 1

        weakest_links, alpha_eff, left_impurity, left_T = self._find_weakest_links(
            node.left, N, weakest_links, alpha_eff
        )
        weakest_links, alpha_eff, right_impurity, right_T = self._find_weakest_links(
            node.right, N, weakest_links, alpha_eff
        )

        this_impurity = node.impurity * node.samples / N
        this_alpha_eff = (this_impurity - (left_impurity + right_impurity)) / (
            left_T + right_T - 1
        )

        if this_alpha_eff < alpha_eff:
            # found new weakest link
            weakest_links = [node]
            alpha_eff = this_alpha_eff
        elif this_alpha_eff == alpha_eff:
            # multiplicity of weakest links
            weakest_links.append(node)

        return (
            weakest_links,
            alpha_eff,
            left_impurity + right_impurity,
            left_T + right_T,
        )

    def predict(self, X):
        """
        Predict values for input observations.

        Parameters
        ----------
        X : (N, p) np.ndarray
            observations

        Returns
        -------
        y_hat : (N,) np.ndarray
            predictions
        """
        y_hat = []
        for x in X:
            y_hat.append(self.root.decide(x))
        return np.asarray(y_hat)

    def node_value(self, y, w):
        """
        Return the node value.

        Parameters
        ----------
        y : (N,) np.ndarray
            targets
        w : (N,) np.ndarray
            sample weight

        Returns
        -------
        val : float or int
            node value
        """
        if self.assign_leaf_node == "most_common":
            unique_classes = np.unique(y)
            w_count = [sum(w[y == k]) for k in unique_classes]
            return unique_classes[np.argmax(w_count)]
        elif self.assign_leaf_node == "mean":
            return np.average(y, weights=w)
        else:
            raise ValueError

    def gini(self, y, w):
        """
        Compute Gini index for a vector of class labels.

        Parameters
        ----------
        y : (N,) np.ndarray
            class labels
        w : (N,) np.ndarray
            sample weight

        Returns
        -------
        Q : float
            Gini index
        """
        if len(y) == 0:
            return 0.0

        Q_tau = 1
        for k in np.unique(y):
            p_tauk = sum(w[y == k]) / sum(w)
            Q_tau -= p_tauk**2
        return Q_tau

    def entropy(self, y, w):
        """
        Compute entropy for a vector of class labels.

        Parameters
        ----------
        y : (N,) np.ndarray
            class labels
        w : (N,) np.ndarray
            sample weight

        Returns
        -------
        Q : float
            entropy
        """
        if len(y) == 0:
            return 0.0

        Q_tau = 0
        for k in np.unique(y):
            p_tauk = sum(w[y == k]) / sum(w)
            Q_tau -= p_tauk * np.log(p_tauk)
        return Q_tau

    def misclassification_rate(self, y, w):
        """
        Compute misclassification rate for a vector of class
        labels.

        Parameters
        ----------
        y : (N,) np.ndarray
            class labels
        w : (N,) np.ndarray
            sample weight

        Returns
        -------
        Q : float
            misclassification rate
        """
        if len(y) == 0:
            return 0.0

        w_prob = [sum(w[y == k]) / sum(w) for k in np.unique(y)]
        return sum(sorted(w_prob)[:-1])

    def squared_error(self, y, w):
        """
        Compute the squared error.

        Parameters
        ----------
        y : (N,) np.ndarray
            targets
        w : (N,) np.ndarray
            sample weight

        Returns
        -------
        Q : float
            squared error
        """
        if len(y) == 0:
            return 0.0

        return np.average((y - np.average(y, weights=w)) ** 2, weights=w)

    def print_tree(self):
        """
        Print the tree structure.
        """

        def _tree_to_str(node, indent="\t"):
            output = "pred={}, impurity={:.5f}, samples={}".format(
                node.value, node.impurity, node.samples
            )
            if not (node.left or node.right):
                return output
            decision = ", x[{}] < {} ?".format(
                node.col_idx,
                node.threshold,
            )
            true_branch = indent + "yes -> " + _tree_to_str(node.left, indent + "\t")
            false_branch = indent + "no  -> " + _tree_to_str(node.right, indent + "\t")
            return output + decision + "\n" + true_branch + "\n" + false_branch

        str_tree = _tree_to_str(self.root)
        print(str_tree)
