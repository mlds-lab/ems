
import numpy as np
from numpy import linalg as LA
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
# from MLE.pySGL import blockwise_descent
from MLE.pyPQN.minConF_PQN import minConf_PQN
from MLE.pyPQN.SquaredError import SquaredError
from MLE.pyPQN.projectRandom2 import randomProject
from MLE.PqnLasso import PqnLassoNet


class TorchLinearRegression:
    def __init__(self, marker_groups, alpha=None, lbda=None):
        """
        Wrapper for Torch Linear Regression model combining  Torch Group Lasso (using PQN) and Ridge penalty.
        This wrapper is compatible with GridSearchCV of Sci-Kit-Learn. Main methods are fit and predict.

        Args:
            marker_groups (list): marker indicators for each feature/column of X
            alpha (int): Ridge regularizer
            lbda (int): Group Lasso regularizer 
        """
        self.marker_groups = marker_groups
        self.alpha = alpha
        self.lbda = lbda
        self.model = None
        self.feature_support = None
        self.group_lasso_strategy = 'PQN'  # or 'SGL'
        self.coef = None

    def fit(self, X, y, sample_weight = None):

        feature_support = self.group_lasso_PQN(X, y, sample_weight)

        model = Ridge(alpha=self.alpha)
        model.fit(X[:, feature_support], y, sample_weight=sample_weight)
        self.coef = model.coef_
        self.model = model
        self.feature_support = feature_support

        return self

    def score(self, X, y, sample_weight = None):

        self.fit(X, y, sample_weight)
        y_predict = self.predict(X)
        score_ = mean_squared_error(y, y_predict)

        return score_

    def get_params(self, deep=True):

        params = {"alpha": self.alpha, "lbda": self.lbda, "marker_groups": self.marker_groups}

        return params

    def set_params(self, **params):

        self.alpha = params["alpha"]
        self.lbda = params["lbda"]
        #self.marker_groups = params["marker_groups"]

        return self

    def predict(self, X):

        model = self.model
        feature_support = self.feature_support
        y_predict = model.predict(X[:, feature_support])

        return y_predict


    def group_lasso_PQN(self, X, y, sample_weight):

        d1, d2 = X.shape
        markers, repeats = np.unique(self.marker_groups, return_counts=True)
        #n_markers = d2 // self.n_bins
        gr = np.asarray(self.marker_groups)#np.zeros(d2)
        #for i in range(n_markers):
        #    gr[i * self.n_bins: (i + 1) * self.n_bins] = i
        net =  PqnLassoNet (layers = [d2] ,groups = gr, reg = 'group', lossFnc ='mse', useProjection = True, lbda = self.lbda)
        w2 = net.fit(X,y, sample_weight)

        feature_support = []
        for j in markers:
            indices = np.where(self.marker_groups == j)[0]
            if LA.norm(w2[indices], 2) != 0:
                feature_support.append(int(j))

        return feature_support


def get_indices(feature_support, n_bins):
    """The function receives the marker indices (and # of bins) as input and returns the expanded feature indices
    in the actual data case."""

    f1 = lambda fs, n_bins1: [list(range(n * n_bins1, (n + 1) * n_bins1)) for n in fs]
    f = lambda fs, n_bins1: sum(f1(fs, n_bins1), [])

    return f(feature_support, n_bins)

