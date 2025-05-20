# SPDX-License-Identifier: BSD-3-Clause
# Adapted for kernel comparison including the Monster kernel

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load iris dataset and use the first two features for 2D visualization
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# Split into train/test for accuracy measurement
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Monster kernel: simulate projection into 196,883D space
rng = np.random.RandomState(42)
D = 196883
W = rng.randn(2, D)

def monster_kernel(X1, X2):
    X1_proj = np.dot(X1, W)
    X2_proj = np.dot(X2, W)
    return np.dot(X1_proj, X2_proj.T)

# SVM regularization parameter
C = 1.0

# Define models
models = [
    ("SVC with linear kernel", svm.SVC(kernel="linear", C=C)),
    ("LinearSVC (linear kernel)", svm.LinearSVC(C=C, max_iter=10000)),
    ("SVC with RBF kernel", svm.SVC(kernel="rbf", gamma=0.7, C=C)),
    ("SVC with polynomial (degree 3)", svm.SVC(kernel="poly", degree=3, gamma="auto", C=C)),
    ("SVC with Monster kernel", svm.SVC(kernel=monster_kernel, C=C))
]

# Set-up 2x3 grid for plotting
fig, sub = plt.subplots(2, 3, figsize=(15, 10))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]

# Evaluate and plot each model
for (title, clf), ax in zip(models, sub.flatten()):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="predict",
        cmap=plt.cm.coolwarm,
        alpha=0.8,
        ax=ax,
        xlabel=iris.feature_names[0],
        ylabel=iris.feature_names[1],
    )
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(f"{title}\nAccuracy: {acc:.2f}")

# Remove unused subplot if number of models < 6
if len(models) < 6:
    for ax in sub.flatten()[len(models):]:
        fig.delaxes(ax)

plt.show()

