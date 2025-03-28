# Custom-Multi-Class-Classification

This repository compared two custom multi-class Logistic Regression implementations with gradient descent solver to Scikit-learn's 
multi-class logistic regression model with optimized newton-cg solver with l2 regularization.

The comparison is done on the dummy wine dataset, which is made up of 13 features, 178 observations, and 3 classes, and 
is accessed through the sklearn.datasets package.

# Usage

Run the evaluation script:

    python evaluate.py

Expected outputs:
- A line plot showcasing Cross Entropy Loss value during training as a function of iteration for the custom Softmax Regression model. 
Both the Softmax Regression and OneVsRest classifiers required a high number of iterations for the loss function to be close to converging and for 
an appropriate accuracy prediction on the test set to be achieved. This could be because a simple gradient descent solver was used 
with a constant learning rate and pseudo-random weight initializations.
- Three scatter plots showcasing the predicted classification of the samples in the test set with the actual class of 
each sample represented by the color of the points. Accuracy is displayed in the top-left corner of the plots. Each 
scatter plot refers to one of the three models:
  - Custom Softmax Regression model where cross entropy loss was used as the cost function and gradient descent as the solver
  - Custom One-vs-Rest Classifier where the custom softmax regression algorithm was used in the binary classifications
  - Scikit-learn's multi-class logistic regression model with newton-cg solver and l2 regularization

# Results

- The Cross Entropy Loss function as a function of iterations during training for the custom Softmax Regression model
    ![Image](https://github.com/user-attachments/assets/e0bdc5c9-ec85-48d1-b2fb-8adbca09b0d5)
- These are the three scatter plots showcasing the performance of three models
    ![Image](https://github.com/user-attachments/assets/b67a6bcc-9924-4f8f-a78e-1fc36af93a35)

# Repository Structure

This repository contains:

    custom_multiclass_classifier.py: Implementation of the SoftmaxRegression and OneVsRestRegression classes that 
    inherit from the CustomMultiClassClassifier parent class
    evaluate.py: Main script for performing comparisons on the dummy wine dataset and generating plots
    requirements.txt: List of required Python packages

Python 3.12 version was used
