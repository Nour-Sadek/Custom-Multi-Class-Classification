import numpy as np
import matplotlib.pyplot as plt


class CustomMultiClassClassifier:

    """A custom multi-class classifier that """

    _learning_rate: float
    _iterations: int
    _accuracy: list[float]

    def __init__(self, learning_rate: float, iterations: int) -> None:
        """Initialize a new CustomMultiClassClassifier instance with _learning_rate <learning_rate> and _iterations
        <iterations>."""
        if learning_rate <= 0:
            raise ValueError("learning rate should be greater than 0")
        if int(iterations) != iterations or iterations <= 0:
            raise ValueError("iterations should be a positive non-zero integer")

        self._learning_rate = learning_rate
        self._iterations = iterations
        self._accuracy = []

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
    def accuracy(self) -> list[float]:
        return self._accuracy

    @property
    def loss(self) -> list[float]:
        return self._loss

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        pass

    def predict(self, X:np.ndarray) -> np.ndarray:
        pass

    def determine_accuracy(self, y: np.ndarray, y_predictions: np.ndarray) -> float:
        """Return the accuracy of the prediction <y_predictions> in comparison to the actual classification <y>."""
        comparison = (y == y_predictions)
        return float(comparison.sum() / y.size)


class SoftmaxRegression(CustomMultiClassClassifier):

    """Using the Softmax algorithm to perform multi-class Logistic Regression Classification"""

    _weights: np.ndarray
    _bias: np.ndarray
    _loss: list[float]

    def __init__(self, learning_rate: float, iterations: int) -> None:
        """Initialize a new SoftmaxRegression instance with _learning_rate <learning_rate> and _iterations
        <iterations>."""
        super().__init__(learning_rate, iterations)
        self._weights = None
        self._bias = None
        self._loss = []

    def softmax(self, X_per_class: np.ndarray) -> np.ndarray:
        """Return the output of applying the softmax function on the <X_by_class> array of shape
        nb_observations x nb_classes where the weights assigned to each have already been applied to their
        respective features."""
        X_per_class_stable = X_per_class - np.max(X_per_class, axis=1, keepdims=True)
        probs_per_class = np.exp(X_per_class_stable) / np.sum(np.exp(X_per_class_stable), axis=1, keepdims=True)
        return probs_per_class

    def cross_entropy_loss(self, design_matrix: np.ndarray, probs_per_class: np.ndarray) -> float:
        """Return the cross entropy loss value between the expected <design_matrix> probabilities and the predicted
        <probs_per_class> probabilities of class affiliation where both arrays are of shape n_observations x n_classes"""
        # 1e-20 is added to prevent the possibility of encountering a np.log(0) value
        return float(- (1 / design_matrix.shape[0]) * np.sum(design_matrix * np.log(probs_per_class + 1e-120)))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Update the <self._weights> and <self._bias> terms trained on the training set <X_train> of shape
        n_observations x n_features and the target variables for each observation <y_train> using Cross Entropy as the
        cost function and Gradient Descent as the optimization algorithm."""
        # Set up initial values of <self._weights> and <self._bias>
        nb_observations, nb_features = X_train.shape
        nb_classes = np.unique(y_train).size
        np.random.seed(20)
        self._weights = np.random.randn(nb_features, nb_classes) * 0.01
        self._bias = np.zeros((1, nb_classes))

        # Create the design matrix for <X_train>
        design_matrix = np.eye(nb_classes)[y_train]

        # If training the model again, remove saved information from previous run
        self._accuracy = []
        self._loss = []

        for _ in range(self._iterations):
            # Apply the softmax regression function
            X_per_class = (X_train @ self._weights) + self._bias
            probs_per_class = self.softmax(X_per_class)

            # Update the weights and bias terms for each class
            d_weights = (1 / nb_observations) * (X_train.T @ (probs_per_class - design_matrix))
            self._weights = self._weights - self._learning_rate * d_weights
            d_bias = (1 / nb_observations) * (np.sum(probs_per_class - design_matrix, axis=0, keepdims=True))
            self._bias = self._bias - self._learning_rate * d_bias

            # Compute the loss and accuracy
            loss = self.cross_entropy_loss(design_matrix, probs_per_class)
            accuracy = self.determine_accuracy(y_train, np.argmax(probs_per_class, axis=1))
            self._accuracy.append(accuracy)
            self._loss.append(loss)

    def predict_probabilities(self, X_test: np.ndarray) -> np.ndarray:
        """Return a 1-D numpy array of probabilities after applying the linear model on <X_test> and transforming
        the values using the sigmoid function"""

        X_per_class = (X_test @ self._weights) + self._bias
        probs_per_class = self.softmax(X_per_class)
        return probs_per_class

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return a 1D numpy array assigning predicted classifications based on classes seen during training for the
        observations in <X_test> of shape nb_observations x nb_features"""
        probs_per_class = self.predict_probabilities(X_test)
        return np.argmax(probs_per_class, axis=1)

    def plot_loss(self) -> None:
        plt.plot(self._loss)
        plt.title("Model Cross Entropy Loss")
        plt.ylabel("Cross Entropy loss")
        plt.xlabel("Iteration")
        plt.legend(["train"], loc="upper left")
        plt.show()


class OneVsRestRegression(CustomMultiClassClassifier):

    """Using the One-Vs-Rest Classifier to perform multi-class Logistic Regression Classification"""

    _nb_classes: int
    _onevrest_models: list[SoftmaxRegression]

    def __init__(self, learning_rate: float, iterations: int, nb_classes: int) -> None:
        """Initialize a new OneVsRestRegression instance with _learning_rate <learning_rate> and _iterations
        <iterations>."""
        super().__init__(learning_rate, iterations)
        self._nb_classes = nb_classes
        self._onevrest_models = [SoftmaxRegression(learning_rate, iterations) for _ in range(nb_classes)]

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit <self._nb_classes> Binary Softmax Regression classifiers where it is one class (1) versus all other
        classes (0) on the given <X_train> and <y_train> arrays."""
        for class_number in range(self._nb_classes):
            # Change all classes other than <class_number> to 0
            y_train_binary = (y_train == class_number).astype(int)
            # Train each binary model using Softmax Regression
            self._onevrest_models[class_number].fit(X_train, y_train_binary)

    def predict_probabilities(self, X_test: np.ndarray) -> np.ndarray:
        """Return a 2D numpy array of shape nb_observations x <self._nb_classes> where the probability for an
        observation to be classified as one class is determined by the Binary Classifiers stored in
        <self._onevrest_models>."""
        probs_per_class = np.zeros((X_test.shape[0], self._nb_classes))
        for class_number in range(self._nb_classes):
            probs_per_class[:, class_number] = self._onevrest_models[class_number].predict_probabilities(X_test)[:, 1]
        return probs_per_class

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return a 1D numpy array assigning predicted classifications based on classes seen during training for the
        observations in <X_test> of shape nb_observations x nb_features. Those predictions are determined by choosing
        the class with the highest probability among all of them per observation."""
        probs_per_class = self.predict_probabilities(X_test)
        # Select the class with the highest probability among all of them per observation
        return np.argmax(probs_per_class, axis=1)


class OneVsOneRegression(CustomMultiClassClassifier):

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        return

