from nltk import ConfusionMatrix
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')
# fetch_openml tries to give data in a pandas dataframe by default, hence the False
X, y = mnist.data, mnist.target
print(X.shape)
print(y.shape)

print(X[0]) # 784x1 array
reshaped = X[0].reshape(28,28)

import matplotlib.pyplot as plt
plt.imshow(reshaped, cmap='gray')

print(y[0])

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

from sklearn.linear_model import LogisticRegression
sm_mod = LogisticRegression(multi_class='multinomial',
			      penalty='l2',
			      C=50,
			      solver='sag',
			      tol=.001,
			      max_iter=1000
			      ).fit(X_train, y_train)
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
print(f'Train Accuracy: {accuracy_score(sm_mod.predict(X_train), y_train)}')
print(f'Test Accuracy: {accuracy_score(sm_mod.predict(X_test), y_test)}')
print("Confusion Matrix: \n"+str(confusion_matrix(y_test, sm_mod.predict(X_test))))
