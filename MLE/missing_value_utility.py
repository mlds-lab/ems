
import numpy as np
from numpy import linalg as LA
from sklearn.preprocessing import Imputer


class ImputerSKL:

    def __init__(self, missing_values='NaN', strategy='mean', model=None):
        """
        Wrapper for SKL Imputer for imputing missing values.

        Args:
            missing_values: missing values used in the data; either 'NaN' or 0
            strategy: strategy used to fit missing values
            model: missing value model found after fit; to be used for predict
        """

        self.missing_values = missing_values
        self.strategy = strategy
        self.model = model

    def fit(self, X):

        p = Imputer(missing_values=self.missing_values, strategy=self.strategy, axis=0)
        X1 = p.fit_transform(X)
        self.model = p

        return X1

    def predict(self, X):

        p = self.model
        X1 = p.transform(X)

        return X1


class ALS:

    def __init__(self, A=None, B=None, T=100, lambda1=0.01, k=10, missing_values='NaN'):
        """
        Alternating Least Squares (ALS) for imputing missing values.

        Args:
            A (numpy array): first factor of shape (d1, k)
            B (numpy array): second factor of shape (d2, k)
            T: number of ALS iterations
            lambda1: ridge regularzer used in LS
            k: rank of the model
            missing_values: missing values used in the data; either 'NaN' or 0
        """

        self.A = A
        self.B = B
        self.k = k
        self.lambda1 = lambda1
        self.T = T
        self.missing_values = missing_values

    def fit(self, X):

        k = self.k
        T = self.T
        lambda1 = lambda2 = self.lambda1

        d1, d2 = np.shape(X)

        if self.missing_values == 'NaN':
            y = X[~np.isnan(X)]
            design = np.zeros((2, len(y)), dtype='int')
            design[0, :], design[1, :] = np.where(~np.isnan(X))
        else:  # missing_value == 0
            y = X[np.nonzero(X)]
            design = np.zeros((2, len(y)), dtype='int')
            design[0, :], design[1, :] = np.nonzero(X)

        # initialize factors A and B:
        A = np.random.uniform(0, 1, (d1, k))
        B = np.random.uniform(0, 1, (d2, k))

        for t in range(T):

            for u in range(d1):

                # 1st LS: estimating user features A:
                user_index = np.where(design[0, :] == u)[0]
                item_index = design[1, user_index]
                XtX_lamb = np.dot(B[item_index, :].T, B[item_index, :]) + lambda1 * np.identity(k)
                XtY = np.dot(B[item_index, :].T, y[user_index])
                A[u, :] = LA.solve(XtX_lamb, XtY).reshape(k, )

            for m in range(d2):

                # 2nd LS: estimating movie features B:
                item_index = np.where(design[1, :] == m)[0]
                user_index = design[0, item_index]
                ZtZ_lamb = np.dot(A[user_index, :].T, A[user_index, :]) + lambda2 * np.identity(k)
                ZtY = np.dot(A[user_index, :].T, y[item_index])
                B[m, :] = LA.solve(ZtZ_lamb, ZtY).reshape(k, )

        self.A = A
        self.B = B

        X1 = np.dot(A, B.T)

        return X1

    def fold_in(self, x):

        B = self.B
        lambda1 = self.lambda1
        k = self.k

        if self.missing_values == 'NaN':
            y = x[~np.isnan(x)]
            items = np.where(~np.isnan(x))[0]
        else:   # missing_value == 0
            y = x[np.nonzero(x)]
            items = np.nonzero(x)[0]

        XtX_lamb = np.dot(B[items, :].T, B[items, :]) + lambda1 * np.identity(k)
        XtY = np.dot(B[items, :].T, y)
        x1 = LA.solve(XtX_lamb, XtY).reshape(k, )

        return x1

    def predict(self, X):

        d1, d2 = np.shape(X)
        X1 = np.zeros((d1, self.k))

        for i in range(d1):
            X1[i, :] = self.fold_in(X[i, :].squeeze())

        B = self.B

        return np.dot(X1, B.T)
