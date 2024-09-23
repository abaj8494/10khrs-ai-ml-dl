# this is a program to iteratively solve a classification problem
import numpy as np

# example = [ bias, x1, x2, class]
a = np.array([1, 0, 1, -1])
b = np.array([1, 2, 0, -1])
c = np.array([1, 1, 1, +1])
X = np.array([a ,b ,c])
eta = 1.0

# by hand we can construct the line 2x_1 + 4x_2 - 5 = 0
# but running the perceptron algorithm by hand is annoying, hence this program:

w = np.array([-1.5, 0, 2])

errors = 3
while errors > 0:
    errors = 3
    for e in X:
        # calculate s
        s = e[0]*w[0] + e[1]*w[1] + e[2]*w[2]
        if s*e[3] > 0: # checks that sign is the same
            errors -= 1
        else: # updates that weight
            if s > 0:
                #subtract
                w = np.array([i-eta*e[j] for j,i in enumerate(w)])
            else:
                #add
                w = np.array([i+eta*e[j] for j,i in enumerate(w)])
        print(w)


# notes:
# you could use a dotproduct
# you could use slicing



