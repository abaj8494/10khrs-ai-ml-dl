# SPDX-License-Identifier: BSD-3-Clause
# Adapted for Monster Kernel demo

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, svm
from sklearn.inspection import DecisionBoundaryDisplay

# Load iris dataset (2D slice for easy visualization)
iris = datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target

# Use fixed random state for reproducibility
rng = np.random.RandomState(42)

# Simulate a projection to a high-dimensional space (e.g., 196,883D)
# We'll fake it via an inner product using a random matrix M: X M M^T Y^T
D = 196883  # Monster dimensionality
W = rng.randn(2, D)  # Project from 2D to 196,883D


def monster_kernel(X1, X2):
    """
    Custom kernel simulating projection to 196,883D:
        k(X, Y) = (X @ W) @ (Y @ W).T
    This gives the same result as a dot product in high-D space.
    """
    X1_proj = np.dot(X1, W)
    X2_proj = np.dot(X2, W)
    return np.dot(X1_proj, X2_proj.T)


# Fit an SVM with the custom Monster kernel
clf = svm.SVC(kernel=monster_kernel)
clf.fit(X, Y)

# Plot decision boundaries
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    cmap=plt.cm.Paired,
    ax=ax,
    response_method="predict",
    plot_method="pcolormesh",
    shading="auto",
)

# Plot training data
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors="k")
plt.title("SVM with Monster Kernel (196,883D Random Projection)")
plt.axis("tight")
plt.show()

