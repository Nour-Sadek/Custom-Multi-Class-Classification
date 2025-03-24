from custom_multiclass_classifier import SoftmaxRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)


# Center the train and test sets
def center_data(X: np.ndarray) -> np.ndarray:
    """Return a modified numpy array where <X> is centered"""
    cols_mean = np.mean(X, axis=0)
    cols_mean_mat = cols_mean * np.ones((X.shape[0], X.shape[1]))
    centered_data = X - cols_mean_mat
    return centered_data


# Center the train and test datasets separately
X_train, X_test = center_data(X_train), center_data(X_test)

"""Output of the custom model"""
custom_model = SoftmaxRegression(0.0001, 25000)
custom_model.fit(X_train, y_train)
custom_model.plot_loss()

# Using the trained model to predict classifications of the test dataset
y_predictions_custom = custom_model.predict(X_test)
y_predicted_probabs_custom = custom_model.predict_probabilities(X_test)
y_probabilities_custom = y_predicted_probabs_custom.argmax(axis=1)

# Sort <y_probabilities_custom> based on increasing order of predicted probabilities
indices_sorted_on_probability = np.argsort(y_probabilities_custom)
y_probabilities_custom_sorted = y_probabilities_custom[indices_sorted_on_probability]
y_test_sorted_based_on_custom_predictions = y_test[indices_sorted_on_probability]  # Apply the same order

# Visualize the probabilities of <X_test> based on predictions of custom model
fig, ax = plt.subplots()
scatter = ax.scatter(range(len(y_probabilities_custom_sorted)), y_probabilities_custom_sorted, linewidths=0.3,
                     edgecolors="w", c=y_test_sorted_based_on_custom_predictions, cmap="rainbow", s=20)
plt.gca().set_xticklabels([])
plt.axhline(y=0.5, color="k", linestyle="--")
handles, labels = scatter.legend_elements()
legend = ax.legend(handles=handles, labels=["0", "1", "2"], title="Actual Class")
plt.text(32, 0, "Accuracy = " + str(round(custom_model.determine_accuracy(y_test, y_predictions_custom), 3)))
plt.ylabel("Model prediction of probability of belonging to either class")
plt.title("Prediction of custom logistic regression model using Gradient Descent")
plt.show()


"""Output of the sci-kit learn LogisticRegression model"""
sklearn_model = LogisticRegression(penalty="l2", solver="newton-cg", random_state=20)
sklearn_model.fit(X_train, y_train)

# Using sci-kit learn's model to predict classifications of the test dataset
y_predictions_sklearn = sklearn_model.predict(X_test)
y_predicted_probabs = sklearn_model.predict_proba(X_test)
y_probabilities_sklearn = y_predicted_probabs.argmax(axis=1)

# Sort <y_probabilities_custom> based on increasing order of predicted probabilities
indices_sorted_on_probability_sklearn = np.argsort(y_probabilities_sklearn)
y_probabilities_sklearn_sorted = y_probabilities_sklearn[indices_sorted_on_probability_sklearn]
y_test_sorted_based_on_sklearn_predictions = y_test[indices_sorted_on_probability_sklearn]  # Apply the same order

# Visualize the probabilities of <X_test> based on predictions of custom model
fig, ax = plt.subplots()
scatter = ax.scatter(range(len(y_probabilities_sklearn_sorted)), y_probabilities_sklearn_sorted, linewidths=0.3,
                     edgecolors="w", c=y_test_sorted_based_on_sklearn_predictions, cmap="rainbow", s=20)
plt.gca().set_xticklabels([])
plt.axhline(y=0.5, color="k", linestyle="--")
handles, labels = scatter.legend_elements()
legend = ax.legend(handles=handles, labels=["0", "1", "2"], title="Actual Class")
plt.text(32, 0, "Accuracy = " + str(round(sklearn_model.score(X_test, y_test), 3)))
plt.ylabel("Model prediction of probability of belonging to either class")
plt.title("Prediction of sci-kit learn's logistic regression model using the newton-cg model with l2 regularization")
plt.show()
