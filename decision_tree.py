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
        """
        self.root = None
        self.criterion = getattr(self, criterion)
        self.assign_leaf_node = assign_leaf_node
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.ccp_alpha = ccp_alpha

    def fit(self, X, y):
        """
        Fit the decision tree to data. First grow the tree
        and optionally prune it.

        Parameters
        ----------
        X : (N, p) np.ndarray
            observations
        y : (N,) np.ndarray
            class labels

        Returns
        -------
        model : DecisionTree
            fitted decision tree
        """
        self.root = self._grow_tree(X, y)
        if self.ccp_alpha > 0:
            self._minimal_cost_complexity_pruning(len(y))
        return self

    def _grow_tree(self, X, y, depth=0):
        """
        Depth-first strategy to build tree greedily.

        Parameters
        ----------
        X : (N, p) np.ndarray
            observations
        y : (N,) np.ndarray
            class labels
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

        impurity_base = self.criterion(y)

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
                self.node_value(y),
                impurity_base,
                len(y),
            )
        else:  # build tree recursively
            for column_i in np.arange(X.shape[1]):
                split_vals = (
                    np.unique(X[:, column_i])[:-1] + np.unique(X[:, column_i])[1:]
                ) / 2
                for split_val in split_vals:
                    y_left = y[X[:, column_i] < split_val]
                    y_right = y[X[:, column_i] >= split_val]
                    # getting the left and right losses
                    impurity_left = self.criterion(y_left)
                    impurity_right = self.criterion(y_right)
                    # calculating the weights for each of the nodes
                    impurity_split = (
                        len(y_left) * impurity_left + len(y_right) * impurity_right
                    ) / len(y)
                    column_i_gain = impurity_base - impurity_split
                    if column_i_gain > best_gain:
                        best_gain = column_i_gain
                        best_column_index = column_i
                        best_column_threshold = split_val

            if best_gain <= 0:
                # stop recursion when there is no more gain
                return TreeNode(
                    None, None, None, None, self.node_value(y), impurity_base, len(y)
                )

            # divide dataset and continue recursion
            X_left = X[X[:, best_column_index] < best_column_threshold]
            y_left = y[X[:, best_column_index] < best_column_threshold]
            X_right = X[X[:, best_column_index] >= best_column_threshold]
            y_right = y[X[:, best_column_index] >= best_column_threshold]

            left_tree = self._grow_tree(X_left, y_left, depth=depth + 1)
            right_tree = self._grow_tree(X_right, y_right, depth=depth + 1)

            return TreeNode(
                left_tree,
                right_tree,
                best_column_index,
                best_column_threshold,
                self.node_value(y),
                impurity_base,
                len(y),
            )

    def _minimal_cost_complexity_pruning(self, N):
        """
        Minimal cost-complexity pruning, implemented
        through iterative weakest-link pruning. For
        a description of the algorithm see
        >> Breiman, L. "Classification and regression trees." Routledge, 2017

        Parameters
        ----------
        N : int
            total number of observations
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
        return np.array([self.root.decide(x) for x in X])

    def node_value(self, y):
        """
        Return the node value.

        Parameters
        ----------
        y : (N,) np.ndarray
            targets

        Returns
        -------
        val : float or int
            node value
        """
        if self.assign_leaf_node == "most_common":
            values, counts = np.unique(y, return_counts=True)
            return values[np.argmax(counts)]
        elif self.assign_leaf_node == "mean":
            return np.mean(y)
        else:
            raise ValueError

    def gini(self, y):
        """
        Compute Gini index for a vector of class labels.

        Parameters
        ----------
        y : (N,) np.ndarray
            class labels

        Returns
        -------
        Q : float
            Gini index
        """
        if len(y) == 0:
            return 0.0

        Q_tau = 1
        for k in np.unique(y):
            p_tauk = np.mean(y == k)
            Q_tau -= p_tauk**2
        return Q_tau

    def entropy(self, y):
        """
        Compute entropy for a vector of class labels.

        Parameters
        ----------
        y : (N,) np.ndarray
            class labels

        Returns
        -------
        Q : float
            entropy
        """
        if len(y) == 0:
            return 0.0

        Q_tau = 0
        for k in np.unique(y):
            p_tauk = np.mean(y == k)
            Q_tau -= p_tauk * np.log(p_tauk)
        return Q_tau

    def squared_error(self, y):
        """
        Compute the mean squared error.

        Parameters
        ----------
        y : (N,) np.ndarray
            targets

        Returns
        -------
        Q : float
            squared error
        """
        if len(y) == 0:
            return 0.0

        return np.mean((y - np.mean(y)) ** 2)

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
