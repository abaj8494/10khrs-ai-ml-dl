import numpy as np
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
#X = iris.data[:, :2]
"""results:
    SVC with linear kernel                   Accuracy: 0.80
    LinearSVC (linear kernel)                Accuracy: 0.78
    SVC with RBF kernel                      Accuracy: 0.80
    SVC with polynomial (degree 3)           Accuracy: 0.78
    SVC with Monster kernel                  Accuracy: 0.82
"""

X = iris.data[:, :3]
"""results:
    SVC with linear kernel                   Accuracy: 1.00
    LinearSVC (linear kernel)                Accuracy: 0.98
    SVC with RBF kernel                      Accuracy: 1.00
    SVC with polynomial (degree 3)           Accuracy: 0.96
    SVC with Monster kernel                  Accuracy: 0.91
"""
#X = iris.data
#1.00 accuracy on all methods

y = iris.target

# train / test split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# random number generator
rng = np.random.RandomState(42)
D = 196883
W = rng.randn(X.shape[1], D)  # Project from 4D to 196,883D

def monster_kernel(X1, X2):
    X1_proj = np.dot(X1, W)
    X2_proj = np.dot(X2, W)
    return np.dot(X1_proj, X2_proj.T)

# Regularization parameter
C = 1.0

# Define models
models = [
    # one vs. one classifier, with dual problem formulation. slower
    ("SVC with linear kernel", svm.SVC(kernel="linear", C=C)),
    # one vs. rest. primal, faster.
    ("LinearSVC (linear kernel)", svm.LinearSVC(C=C, max_iter=10000)),
    ("SVC with RBF kernel", svm.SVC(kernel="rbf", gamma=0.7, C=C)),
    ("SVC with polynomial (degree 3)", svm.SVC(kernel="poly", degree=3, gamma="auto", C=C)),
    ("SVC with Monster kernel", svm.SVC(kernel=monster_kernel, C=C))
]

# Train, predict, and print accuracy
print("Classification Accuracy:\n")
for name, clf in models:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name:<40} Accuracy: {acc:.2f}")

