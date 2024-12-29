import numpy as np
import numpy.linalg as linalg
import util
import matplotlib.pyplot as plt



def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)

    print("debug:", x_train.shape, y_train.shape)
    print("theta 0", np.zeros(x_train.shape[1]).reshape(-1,3).shape)
    # *** START CODE HERE ***
    # Train a logistic regression classifier
    clf = LogisticRegression(theta_0=np.zeros(x_train.shape[1]).T,eps=0.00001)
    clf.fit(x_train, y_train)
    clf.predict(x_valid)
    # Plot decision boundary on top of validation set set
    colors = np.array(['orange','blue'])
    random_labelling = np.random.choice([0,1],size=x_train.shape[0])
    plt.scatter(x_valid[:,1], x_valid[:,2], c=colors[y_valid.astype(int)])
    f = lambda x: x**2
    x_values = np.array([min(x_train[:, 1]), max(x_train[:, 1])])
    y_values = -(clf.theta[0] + clf.theta[1] * np.array(x_values)) / clf.theta[2]
    plt.plot(x_values, y_values, label='Decision Boundary')
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.savefig("random_labelling.png")
    plt.legend()
    plt.title("dataset 1")
    plt.show()

    # Use np.savetxt to save predictions on eval set to save_path
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, x, y):
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])

        for _ in range(self.max_iter):
            z = np.dot(x, self.theta)
            h = self.sigmoid(z)

            # Gradient
            gradient = np.dot(x.T, h - y)

            # Hessian
            W = np.diag(h * (1 - h))
            H = np.dot(np.dot(x.T, W), x)

            # Update theta
            delta_theta = np.dot(linalg.inv(H), gradient)
            self.theta -= self.step_size * delta_theta

            # Check for convergence
            if np.linalg.norm(delta_theta, ord=1) < self.eps:
                break

    def predict(self, x):
        return self.sigmoid(np.dot(x, self.theta))


    # def fit(self, x, y):
    #     """Run Newton's Method to minimize J(theta) for logistic regression.
    #
    #     Args:
    #         x: Training example inputs. Shape (n_examples, dim).
    #         y: Training example labels. Shape (n_examples,).
    #     """
    #     # *** START CODE HERE ***
    #     # return theta_vec = theta_vec - H^-1 @ del l
    #     # l = nasty
    #     # del l_j = (y-h_theta(x))x_j
    #     # h_theta = sigmoid(theta T x)
    #
    #     def hessian(nabla, theta, x):
    #         d = x.shape[1]
    #         H = np.array((d,d))
    #         for row in range(d):
    #             for col in range(d):
    #                 H[row, col] = -1 * sigmoid(theta.T @ x[row:]) @ (1-sigmoid(theta.T @ x[row:])) * x[row] * x[col]
    #         return H
    #
    #     nabla = np.zeros(3)
    #     for i in range(3):
    #         nabla[i] = (y-sigmoid(self.theta.T @ x[i,:])) @ x[:,i]
    #
    #     for _ in range(self.max_iter):
    #         new_theta = self.step_size * np.matmul(linalg.inv(hessian(nabla,self.theta,x)), nabla)
    #         if self.theta - new_theta < self.eps:
    #             break
    #         else:
    #             self.theta = new_theta
    #     # *** END CODE HERE ***
    #
    # def predict(self, x):
    #     """Return predicted probabilities given new inputs x.
    #
    #     Args:
    #         x: Inputs of shape (n_examples, dim).
    #
    #     Returns:
    #         Outputs of shape (n_examples,).
    #     """
    #     # *** START CODE HERE ***
    #     return self.theta.T @ x
    #     # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')

