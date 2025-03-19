import numpy as np
import matplotlib.pyplot as plt


class CustomMultiClassClassifier:

    """A custom multi-class classifier that """

    _learning_rate: float
    _iterations: int
    _weights: np.ndarray
    _bias: np.ndarray
    _accuracy: list[float]
    _loss: list[float]

    def __init__(self, learning_rate: float, iterations: int) -> None:
        if learning_rate <= 0:
            raise ValueError("learning rate should be greater than 0")
        if int(iterations) != iterations or iterations <= 0:
            raise ValueError("iterations should be a positive non-zero integer")

        self._learning_rate = learning_rate
        self._iterations = iterations
        self._weights = None
        self._bias = None
        self._accuracy = []
        self._loss = []

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def bias(self) -> np.ndarray:
        return self._bias

    @property
    def history(self) -> dict[str: list]:
        return self._history

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        pass

    def predict(self, X:np.ndarray) -> np.ndarray:
        pass

    def accuracy(self, y: np.ndarray, y_predictions: np.ndarray) -> float:
        comparison = (y == y_predictions)
        return float(comparison.sum() / y.size)


class SoftmaxRegression(CustomMultiClassClassifier):

    """Using the Softmax algorithm to perform multi-class Logistic Regression Classification"""

    def softmax(self, X_by_class: np.ndarray) -> np.ndarray:
        """Return the output of applying the softmax function on the <X_by_class> array of shape
        nb_observations x nb_classes where the weights assigned to each have already been applied to their
        respective features."""
        X_per_class_stable = X_by_class - np.max(X_by_class)
        probs_per_class = np.exp(X_per_class_stable) / np.sum(np.exp(X_per_class_stable))
        return probs_per_class

    def cross_entropy_loss(self, design_matrix: np.ndarray, probs_per_class: np.ndarray) -> float:
        """Return the cross entropy loss value between the expected <design_matrix> probabilities and the predicted
        <probs_per_class> probabilities of class affiliation where both arrays are of shape n_observations x n_classes"""
        # 1e-20 is added to prevent the possibility of encountering a np.log(0) value
        return - (1 / design_matrix.shape[0]) * np.sum(design_matrix * np.log(probs_per_class + 1e-120))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


class OneVsRestRegression(CustomMultiClassClassifier):

    def fit(self):
        return


class OneVsOneRegression(CustomMultiClassClassifier):

    def fit(self):
        return

