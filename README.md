<!-- 
This is an implementation of the ExtraTrees algorithm from the paper Extremely Randomized Trees by Gehrts, Ernst and Wehenkel.

The file trees.py contains all the code for single trees and a random forest. To the best of my ability, I recreated the algorithm for classification as outlined in the paper except for the voting mechanism for trees: I used soft voting, 
where each tree returns the probabilities of it's prediction for each class, then the forest prediction takes the average of these probabilities across every tree in the forest and returns the class with the highest probability.

The trees.py file, when run as main, will test an individual tree and a forest on a toy dataset that of normally distributed points in the x-y plane that need to be classified on if the product of the coordinates is positive or negative.
Predictably, the error rate is high if k (the number of features included) is 2 since the ideal first split is to split the dataset in half based on whether x (or y) is positive or negative. But because the data is randomly distributed, this should not decrease
the entropy at all, whereas a more extreme split will do better as the initial feature, but leads to worse overall performance.

Running the iris_test.py file as main tests the accuracy of the algorithm on the iris classification toy data set with an 80-20 training-test split.
-->
# ExtraTrees Implementation

This is an implementation of the **ExtraTrees** algorithm based on the paper *Extremely Randomized Trees* by Geurts, Ernst, and Wehenkel.

## Implementation Details

The file `trees.py` contains all the code for both single trees and a random forest. 

To the best of my ability, I recreated the algorithm for classification as outlined in the paper, with one exception regarding the voting mechanism:

*   **Soft Voting:** Instead of standard voting, I used soft voting. Each tree returns the probabilities of its prediction for each class. The forest prediction then takes the average of these probabilities across every tree and returns the class with the highest probability.

## Usage
### 0. Requirements
This requires `numpy` to run `trees.py`, and both `numpy` and `scikit-learn` for `iris_test.py`

### 1. Toy Dataset (`trees.py`)
When run as main, `trees.py` tests an individual tree and a forest on a generated toy dataset.

*   **Dataset:** Normally distributed points in the x-y plane.
*   **Target:** Classify if the product of the coordinates is positive or negative.

> **Note on Performance:** Predictably, the error rate is high if `k` (the number of features included) is 2. Since the ideal first split is to split the dataset in half based on whether x (or y) is positive or negative, but the data is randomly distributed, this split does not decrease the entropy at all. A more extreme split will do better as the initial feature but leads to worse overall performance.

### 2. Iris Dataset (`iris_test.py`)
Running `iris_test.py` as main tests the accuracy of the algorithm on the standard Iris classification toy dataset using an 80-20 training-test split. The current parameters in the forest predict 29/30 of the test cases correctly.
