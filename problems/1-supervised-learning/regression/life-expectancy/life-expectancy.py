import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.float_format = '{:.2f}'.format
le_df = pd.read_csv("life_expectancy.csv")
le_df.columns

sns.boxplot(x=le_df['Life expectancy '])
plt.show()

#+RESULTS:
[[file:./.ob-jupyter/d19582e2992816c0d64efb4fb9f61227721337d6.png]]

le_df.info()

le_df = le_df.drop(columns=['Country','Status'])

import numpy as np
corr_mat = le_df.corr()
mask = np.zeros_like(corr_mat, dtype=bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(20,19))
cmap=sns.diverging_palette(220,10, as_cmap=True)
sns.heatmap(corr_mat, mask=mask, cmap=cmap, vmax=.3, center=0,
	    square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

#+RESULTS:
[[file:./.ob-jupyter/372f5dd471f27160188f332ddd7fd92cf48ad44e.png]]

sns.pairplot(le_df)
plt.show()

#+RESULTS:
[[file:./.ob-jupyter/c3130272e3c33f0879b8c2bfd8bb955d46495c06.png]]

from sklearn.preprocessing import StandardScaler

le_df_noNAN = le_df.dropna()
X = le_df_noNAN.drop(columns=["Life expectancy "])
y = le_df_noNAN["Life expectancy "].copy()

scaler = StandardScaler().fit(X)
scaled_X = scaler.transform(X)

print(f"Feature Means: {scaled_X.mean(axis=0)}")
print(f"Feature Variances: {scaled_X.var(axis=0)}")

from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=73)
y_train, y_test = train_test_split(y, test_size=0.2, random_state=73)

%%time
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge
lambdas = [0.01, 0.1, 0.5, 1, 1.5, 2, 5, 10, 20, 30, 50, 100, 200, 300]
N = len(lambdas)
coefs_mat = np.zeros((X_train.shape[1], N))
for i in range(N):
  L = lambdas[i]
  ridge_lm = Ridge(alpha=L).fit(X_train, y_train)
  coefs_mat[:,i] = ridge_lm.coef_

plt.figure(figsize=(10,10))
for i in range(X_train.shape[1]):
  lab = "X" + str(i + 1)
  plt.plot(np.log(lambdas), coefs_mat[i], label=lab)
  plt.legend()
plt.xlabel(r"log($\lambda$)")
plt.ylabel("Estimated Coefficient")
plt.show()

#+RESULTS:
:RESULTS:
[[file:./.ob-jupyter/cf38f6d2494c73bb3506e3422b9bf98e5e739bba.png]]
: CPU times: user 939 ms, sys: 275 ms, total: 1.21 s
: Wall time: 256 ms
:END:

%%time
lambdas = np.arange(0,50.1,step=0.1)
n = X_train.shape[0]
N = lambdas.shape[0]
CV_score = np.zeros(N)
curIdx = 0
#X_train = X_train.to_numpy()
#y_train = y_train.to_numpy()
for L in lambdas:
  sq_errs = 0.
  for i in range(100):
    x_i = X_train[i]
    x_removed_i = np.delete(X_train, i, axis=0)
    y_i = y_train[i]
    y_removed_i = np.delete(y_train, i, axis=0)

    mod = Ridge(alpha=L).fit(x_removed_i, y_removed_i)
    sq_errs += (mod.predict(x_i.reshape(1,-1))-y_i)**2

  CV_score[curIdx] = sq_errs/n
  curIdx += 1

min_idx = np.argmin(CV_score)
plt.plot(lambdas, CV_score)
plt.xlabel(r"log($\lambda$)")
plt.ylabel("LOOCV (Ridge)")
plt.axvline(x=lambdas[min_idx], color='red')
plt.annotate(f"$\lambda = {lambdas[min_idx]}$", xy=(25,1800))
plt.show()

#+RESULTS:
:RESULTS:
[[file:./.ob-jupyter/6eaa1aaedaeb15819249479223144bb3c214c4a0.png]]
: CPU times: user 1min 59s, sys: 1.13 s, total: 2min
: Wall time: 25.1 s
:END:

from sklearn.linear_model import Lasso
lambdas = [0.01, 0.1, 0.5, 1, 1.5, 2, 5, 10, 20, 30, 50, 100, 200, 300]
N = len(lambdas)
coefs_mat = np.zeros((X_train.shape[1], N))
for i in range(N):
  L = lambdas[i]
  ridge_lm = Lasso(alpha=L).fit(X_train, y_train)
  coefs_mat[:,i] = ridge_lm.coef_

plt.figure(figsize=(10,10))
for i in range(X_train.shape[1]):
  lab = "X" + str(i + 1)
  plt.plot(np.log(lambdas), coefs_mat[i], label=lab)
  plt.legend()
plt.xlabel(r"log($\lambda$)")
plt.ylabel("Estimated Coefficient")
plt.show()

#+RESULTS:
[[file:./.ob-jupyter/2930a91b19b99354bfa807b2906b45038ed643e9.png]]

%%time
lambdas = np.arange(0,50.1,step=0.1)
n = X_train.shape[0]
N = lambdas.shape[0]
CV_score = np.zeros(N)
curIdx = 0
#X_train = X_train.to_numpy()
#y_train = y_train.to_numpy()
for L in lambdas:
  sq_errs = 0.
  for i in range(20): #note we are not going to N
    x_i = X_train[i]
    x_removed_i = np.delete(X_train, i, axis=0)
    y_i = y_train[i]
    y_removed_i = np.delete(y_train, i, axis=0)

    mod = Lasso(alpha=L).fit(x_removed_i, y_removed_i)
    sq_errs += (mod.predict(x_i.reshape(1,-1))-y_i)**2

  CV_score[curIdx] = sq_errs/n
  curIdx += 1

min_idx = np.argmin(CV_score)
plt.plot(lambdas, CV_score)
plt.xlabel(r"log($\lambda$)")
plt.ylabel("LOOCV (Lasso)")
plt.axvline(x=lambdas[min_idx], color='red')
plt.annotate(f"$\lambda = {lambdas[min_idx]}$", xy=(25,1800))
plt.show()

#+RESULTS:
:RESULTS:
[[file:./.ob-jupyter/5abb08d77fe2e6eaa80d2d3337383a85e60f37c7.png]]
: CPU times: user 8min 24s, sys: 47.3 s, total: 9min 11s
: Wall time: 1min 14s
:END:

from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
m_lassoCV = LassoCV(cv=5).fit(X_train, y_train)
ypred_train_lassoCV = m_lassoCV.predict(X_train)
ypred_test_lassoCV = m_lassoCV.predict(X_test)
print(f"Mean Squared Error (TRAIN): {mean_squared_error(y_train,ypred_train_lassoCV)}")
print(f"Mean Squared Error (TEST) : {mean_squared_error(y_test, ypred_test_lassoCV)}")
print(f"With Lambda as {m_lassoCV.get_params()}")

from sklearn.metrics import r2_score
print(f'Train Accuracy: {r2_score(ypred_train_lassoCV, y_train)}')
print(f'Test Accuracy: {r2_score(ypred_test_lassoCV, y_test)}')
