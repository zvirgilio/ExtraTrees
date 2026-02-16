import numpy as np
from numpy.random import Generator
from collections import namedtuple
from typing import Union
import networkx as nx

class Leaf:
    def __init__(self, probs: dict[int, float]):
        self.probs = probs


class Node:
    def __init__(self, attr: int, split: float, left: Leaf | Node, right: Leaf | Node):
        self.attr = attr
        self.split = split
        self.left = left
        self.right = right


# tree = Union(Node, Leaf)

class ScoredSplit(namedtuple("ScoredSplit", ["feature", "split", "is_left", "score"])):
    feature: int
    split: np.float64
    is_left: np.ndarray
    score: np.float64

    def __lt__(self, other):
        return self.score < other.score

# def print_tree(tree, depth = 10):
#     if isinstance(tree, Leaf):
#         print(tree.value)
#     print(f"Attr: {tree.attr}; Split: {tree.split}")
#     print

# these require handling edge cases possibly if data has 0 entropy
def entropy(Y):
    _, counts = np.unique(Y, return_counts = True)
    probs = counts / len(Y)
    return -(probs * np.log(probs)).sum()

def scaled_information_gain(Y, is_left):

    # Compute the entropy of the split (just split size)
    entropy_split = entropy(is_left)
    # Compute entropy of the classification
    entropy_class = entropy(Y)

    # compute left and right children entropy
    entropy_left = entropy(Y[is_left])
    entropy_right = entropy(Y[~is_left])
    p_left = sum(is_left) / len(is_left)
    info_gain = p_left * entropy_left + (1-p_left) * entropy_right
    return info_gain / (entropy_split + entropy_class)



# build tree
def build_tree(X: np.ndarray, Y: np.ndarray, k: int, n_min: int, rng: Generator) -> Node:
    # X: (features, samples)
    # Y: (1 x samples) correct classes
    # k: features to select from randomly when scoring where to split
    # n_min: minimum number of samples before ending the build process
    #

    assert X.shape[0] > 0, "X must be non-empty"
    assert Y.shape[0] == X.shape[1], "X and Y must have the same number of entries"
    assert k > 0, "k must be positive"
    assert n_min > 0, "n_min must be positive"

    # TODO: Recursion termination conditions
    # 1. check for constant features and do not include them
    # if all features are constant, terminate and return the Leaf
    # 2. if samples < n_min return the leaf (DONE)
    # 3. if all labels are constant         (DONE)
    # breakpoint()
    features = range(X.shape[0])
    split_features = rng.choice(features, min(len(features), k), replace=False)
    scored_splits: list[ScoredSplit] = []
    for i in split_features:
        v_min, v_max = X[i].min(), X[i].max()
        split = rng.uniform(v_min, v_max)
        is_left = (X[i] < split)

        score = scaled_information_gain(Y, is_left)
        scored_splits.append(ScoredSplit(i, split, is_left, score))

    feature, split, is_left, _ = max(scored_splits)

    X_left, X_right = X[..., is_left], X[..., ~is_left]
    Y_left, Y_right = Y[is_left], Y[~is_left]

    left_only_one_label = len(np.unique(Y_left)) <= 1
    left_min_samples = len(Y_left) <= n_min
    left_same_cols = np.all(X_left == X_left[:, 0:1], axis=0)
    left_same_features = np.all(left_same_cols)

    right_only_one_label = len(np.unique(Y_right)) <= 1
    right_min_samples = len(Y_right) <= n_min
    right_same_cols = np.all(X_right == X_right[:, 0:1], axis=0)
    right_same_features = np.all(right_same_cols)


    if left_only_one_label or left_min_samples or left_same_features:
        # breakpoint()
        Y_left_un, Y_left_cts = np.unique(Y_left, return_counts = True)
        tree_left = Leaf({i: j/len(Y_left) for i, j in zip(Y_left_un, Y_left_cts)})
    else:
        tree_left = build_tree(X_left, Y_left, k, n_min, rng)

    if right_only_one_label or right_min_samples or right_same_features:
        Y_right_un, Y_right_cts = np.unique(Y_right, return_counts=True)
        tree_right = Leaf({i: j/len(Y_right) for i, j in zip(Y_right_un, Y_right_cts)})
    else:
        tree_right = build_tree(X_right, Y_right, k, n_min, rng)

    return Node(feature, split, tree_left, tree_right)


def pred_single_tree(tree: Node | Leaf, X: np.ndarray) -> np.ndarray:
    # X should be a single vector of features to classify
    # X: (features) array
    # requires n_min = 1 for tree in order to make sense
    # if n_min > 1, would need a voting mechanism if there are counts
    if isinstance(tree, Leaf):
        return tree.probs

    attr, split = tree.attr, tree.split
    if X[attr] < split:
        return pred_single_tree(tree.left, X)
    else:
        return pred_single_tree(tree.right, X)

def ensemble(X: np.ndarray, Y: np.ndarray, k: int, n_min: int, rng: Generator, n_trees: int) -> list[node | leaf]:
    return [build_tree(X, Y, k, n_min, rng) for _ in range(n_trees)]

def pred_from_ensemble(forest: list[node | leaf], X):
    # use soft voting: each tree returns the prob of each class from its own predictions
    # then average those probs across all trees in the forest and select the class with highest overall prob
    probs = {}
    # breakpoint()
    for tree in forest:
        tree_probs = pred_single_tree(tree, X)
        for c in tree_probs:
            if c not in probs:
                probs[c] = [tree_probs[c]]
            else:
                probs[c].append(tree_probs[c])

    for c in probs:
        probs[c] = sum(probs[c])/len(forest)

    return max(probs, key=probs.get)


if __name__ == '__main__':
    rng = np.random.default_rng(1234)
    # generate array of x,y points
    X = rng.normal(size = (2, 500))
    Y = (X[0] * X[1] > 0).astype(int)
    # data set for predicting if the product of coordinates in the x-y plane is positive (quadrants 1 and 3)
    # X = np.array([
    #     [1.0, 0.8, -0.6, -1.2],
    #     [.95, -.3, 1.5, -3.5]
    # ])
    # Y = np.array([1, 0, 0, 1])
    # rng = np.random.default_rng(1234)
    # breakpoint()
    quadrant_tree = build_tree(X, Y, k=1, n_min=1, rng=rng)
    # breakpoint()
    print(pred_single_tree(quadrant_tree, X[:,1]))
    print(f"The label for {X[:, 1]} is {Y[1]}")
    print(pred_single_tree(quadrant_tree, np.array([1.2, 0.2])))        # 1
    print(pred_single_tree(quadrant_tree, np.array([-0.4, 0.2])))       # 0
    print(pred_single_tree(quadrant_tree, np.array([0.4, -1])))         # 0
    print(pred_single_tree(quadrant_tree, np.array([-0.5, -.86])))      # 1

    forest = ensemble(X, Y, k=1, n_min=4, rng=rng, n_trees=10)
    print(pred_from_ensemble(forest, np.array([1.2, 0.2])))        # 1
    print(pred_from_ensemble(forest, np.array([-0.4, 0.2])))       # 0
    print(pred_from_ensemble(forest, np.array([0.4, -1])))         # 0
    print(pred_from_ensemble(forest, np.array([-0.5, -.86])))      # 1
    X_test = rng.normal(size=(2, 50))
    Y_test = (X_test[0] * X_test[1] > 0).astype(int)
    # breakpoint()
    Y_pred = [pred_from_ensemble(forest, x) for x in np.transpose(X_test)]
    print(f"Accuracy is {int(sum(Y_pred == Y_test))} out of {len(Y_test)}, or {sum(Y_pred==Y_test)/len(Y_test):.2f}")
# compute score for splits
# forest ensemble
# print trees
