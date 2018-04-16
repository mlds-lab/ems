
import numpy as np
from numpy import linalg as LA
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from MLE.pySGL import blockwise_descent
from MLE.pyPQN.minConF_PQN import minConf_PQN
from MLE.pyPQN.SquaredError import SquaredError
from MLE.pyPQN.projectRandom2 import randomProject
# import trace


class LinearRegression:

    def __init__(self, marker_groups, alpha=None, lbda=None):
        """
        Wrapper for Linear Regression model combining Group Lasso (using either PQN or SGL) and Ridge.
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
        self.feature_indices = None
        self.group_lasso_strategy = 'PQN'  # 'PQN' or 'SGL' or 'None'
        self.coef = None
        self.sample_weight = None
        self.X = None
        self.y = None

    def fit(self, X, y, sample_weight=None):

        if sample_weight is None:
            sample_weight = np.ones(y.shape)

        self.sample_weight = sample_weight
        self.X = X
        self.y = y

        # print('-'*20 + "Running group lasso:")
        # print(f'X = {X}, y = {y}, sample_weight = {sample_weight}')

        if self.group_lasso_strategy == 'SGL':
            feature_support = self.group_lasso_SGL(X, y)
        elif self.group_lasso_strategy == 'PQN':
            feature_support = self.group_lasso_PQN(X, y, sample_weight)
        else:
            markers, repeats = np.unique(self.marker_groups, return_counts=True)
            feature_support = markers

        markers, repeats = np.unique(self.marker_groups, return_counts=True)
        feature_indices = []
        for i in markers:
            if i in feature_support:
                feature_indices.extend(np.repeat(True, repeats[i]))
            else:
                feature_indices.extend(np.repeat(False, repeats[i]))

        # print('-'*20 + "Running Ridge:")

        model = Ridge(alpha=self.alpha)
        model.fit(X[:, feature_indices], y, sample_weight=sample_weight)
        self.coef = model.coef_

        # print('-'*20 + "Saving regression results.")

        self.model = model
        self.feature_support = feature_support
        self.feature_indices = feature_indices

        return self

    def score(self, X, y, sample_weight=None):

        self.fit(X, y, sample_weight)
        y_predict = self.predict(X)
        score_ = mean_squared_error(y, y_predict)

        return score_

    def get_params(self, deep=True):

        params = {"alpha": self.alpha, "lbda": self.lbda, "marker_groups": self.marker_groups}

        return params

    def set_params(self, **params):

        for parameter, value in params.items():
            setattr(self, parameter, value)

        return self

    def predict(self, X):

        model = self.model
        feature_indices = self.feature_indices

        if feature_indices is None:
            raise ValueError('Feature indices cannot be None!')

        # print('-' * 20 + "Running predict for linear regression.")
        # print(f'X = {X[:, feature_indices]}')

        y_predict = model.predict(X[:, feature_indices])

        return y_predict

    def group_lasso_SGL(self, X, y):

        markers = np.unique(self.marker_groups)

        fs_model = blockwise_descent.SGL(groups=self.marker_groups, alpha=0., lbda=self.lbda, rtol=1e-3)
        fs_model.fit(X, y)
        coefs = fs_model.coef_

        feature_support = []
        for j in markers:
            indices = np.where(self.marker_groups == j)[0]
            if LA.norm(coefs[indices], 2) != 0:
                feature_support.append(int(j))

        return feature_support

    def group_lasso_PQN(self, X, y, sample_weight):

        d1, d2 = X.shape
        markers, repeats = np.unique(self.marker_groups, return_counts=True)

        # print('-'*20 + 'Running PQN:', d2)

        w1 = np.zeros((d2,))

        # tracer for segfault:
        # tracer = trace.Trace(trace=1, count=0, outfile='trace_output')
        # tracer.run('minConf_PQN(fun_object, w1, fun_project, verbose=3)[0]')
        # tracer.results().write_results(show_missing=True)

        w2 = minConf_PQN(self.fun_object, w1, self.fun_project, verbose=0)[0]


        # print('-' * 20 + 'Producing the feature support after PQN run:')
        # print(f'w2 = {w2}')

        feature_support = []
        for j in markers:
            indices = np.where(self.marker_groups == j)[0]
            if LA.norm(w2[indices], 2) != 0:
                feature_support.append(int(j))

        return feature_support

    def fun_project(self, w):

        # print('-'*20 + 'Starting fun_project')
        markers, repeats = np.unique(self.marker_groups, return_counts=True)


        v = np.zeros((markers.shape[0],))
        v1 = np.zeros(w.shape)

        # print('-' * 20 + 'Starting for loop in fun_project')
        # print(f'markers={markers}')
        # print(f'w={w}')

        for i in markers:
            indices1 = np.where(self.marker_groups == i)[0]
            w_group = w[indices1]
            # print(f'w_group={w_group}')
            v[i] = LA.norm(w_group, 2)

            # print(f'v[i]={v[i]}')

            if v[i] != 0:
                v1[indices1] = w_group / v[i]

        # print('-' * 20 + 'Calling randomProject:')


        p = randomProject(v, self.lbda)

        test = v1 * np.repeat(p, repeats)

        # print('-' * 20 + 'randomProject Finished:', p, repeats)
        # print('-'*20 + f'test={test}')

        return test

    def fun_object(self, w):
        # return SquaredError(w, X, y)

        # print('-' * 20 + 'Running fun_object')

        test = SquaredError(w, np.tile(np.reshape(np.sqrt(self.sample_weight), (np.size(self.X, 0), 1)),
                            (1, np.size(self.X, 1))) * self.X, np.sqrt(self.sample_weight) * self.y)

        # print('-' * 20 + 'Returning from fun_object')

        return test
