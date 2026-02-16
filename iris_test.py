from sklearn.datasets import load_iris
import numpy as np
from trees import ensemble, pred_from_ensemble

iris_X, iris_Y = load_iris(return_X_y=True)


rng = np.random.default_rng(42)
training_samples = rng.choice(np.arange(len(iris_Y)), size=int(0.8*len(iris_Y)), replace=False)
train_mask = np.array([i in training_samples for i in range(len(iris_Y))])

X_train = np.transpose(iris_X[train_mask])
Y_train = iris_Y[train_mask]
forest = ensemble(X_train, Y_train, k=1, n_min = 1, rng=rng, n_trees=10)

Y_pred = [pred_from_ensemble(forest, x) for x in iris_X[~train_mask]]
Y_act = iris_Y[~train_mask]
n_correct = sum(Y_pred == Y_act)
acc = n_correct / len(Y_pred)
print(f"Predicted {n_correct} out of {len(Y_pred)} for an accuracy of {acc:.2f}")
