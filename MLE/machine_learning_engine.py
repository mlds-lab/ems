
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, GroupKFold
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from scipy.stats import pearsonr, sem
from MLE.missing_value_utility import ImputerSKL, ALS
from MLE.pattern_matching_utility import find_pattern, predict_pattern
from MLE.linear_regression import LinearRegression
from MLE.linear_regression_torch import TorchLinearRegression
from MLE.TorchNN import TorchNN


class MachineLearningEngine:

    def __init__(self, models, marker_groups, loss_function="mse", n_selected_features=1, n_folds=5,
                 n_folds_inner=5, n_patterns=3):
        """
        The core of the machine learning algorithms is implemented in the MachineLearningEngine class. This class
        implements a fit/predict interface similar to Sci-Kit-Learn.

        Args:
            models (list): Classification or Regression models defined as a list of dictionaries.
            marker_groups (list): marker group indicators of shape (d2, ).
            loss_function (str): indicates the loss function to be used in cross validation; set to 'mse' by default
            n_selected_features (int): number of features selected in feature selection; set to d2 by default
            n_folds (int): number of cross validation folds for the outer CV loop; set to 5 by default
            n_folds_inner (int): number of cross validation folds for the inner CV loop; set to 5 by default
            n_patterns (int): number of top (most frequent) missing value patterns in the data; set to 3 by default

        """

        self.models = models
        self.trained_model = {}
        self.trained_params = {}
        self.default = None

        # scores across different CV folds:
        self.scores = {}        # MSE
        self.R2_scores = {}     # R2
        self.r2_scores = {}     # r2

        self.y_gen = None   # generalization performance estimations

        # Overall generalization performance scores:
        self.MSE = 0       # MSE (Mean Squared Error)
        self.R2 = 0        # R2 (Pearson Coefficient)
        self.r2 = 0        # r2 (Coefficient of Determination)

        self.loss_function = loss_function
        self.marker_groups = marker_groups

        self.n_selected_features = n_selected_features
        self.feature_support = {}
        self.feature_support_lasso = {}
        self.feature_support_nn_lasso = {}
        self.feature_support_torchNN_lasso = {}

        self.n_folds = n_folds          # number of folds for outer and inner CV loops
        self.n_folds_inner = n_folds_inner
        self.n_patterns = n_patterns    # number of top missing value patterns to fit

        self.missing_value_strategy = 'Imputer'    # 'Imputer' or 'ALS'
        self.completion_model = None
        self.patterns = None            # missing value patterns
        self.feature_selection_strategy = 'GroupLasso'  # 'GroupLasso' or 'Stagewise'
        self.cross_val_strategy = 'groups'

        # cross-validation statistics:
        self.cv_params = np.zeros((n_patterns, len(models), n_folds), dtype=object)
        self.cv_mean_train_scores = np.zeros((n_patterns, len(models), n_folds), dtype=object)
        self.cv_mean_test_scores = np.zeros((n_patterns, len(models), n_folds), dtype=object)
        self.regression_coef = np.zeros((n_patterns, n_folds), dtype=object)
        self.nn_regression_coef = np.zeros((n_patterns, n_folds), dtype=object)

    def fit_submodel(self, X, y, indices=None, p_index=0, q=None, g=None):
        """
        Method for fitting the submodel corresponding to pattern p_index.

        Args:
            X (numpy array): data; each row corresponds to a data instance
            y (numpy array): labels
            indices (numpy array): indices of data cases in the data set
            p_index (int): index of the submodel pattern
            q (numpy array): quality of data instances (optional)
            g (numpy array): groups or subject IDs for data instances (optional)

        Returns:
            scores (numpy array): array of MSE test scores over different CV folds
            trained_model (numpy array): array of trained model objects used for different CV folds
            trained_params (numpy array): array of optimal parameters used for different CV folds
        """

        models = self.models

        if g is None:
            g = np.arange(X.shape[0])
        if q is None:
            q = np.ones((X.shape[0], ))

        scores = np.zeros((len(models), self.n_folds))
        R2 = np.zeros((len(models), self.n_folds))
        r2 = np.zeros((len(models), self.n_folds))

        trained_model = np.zeros((len(models), self.n_folds), dtype=object)
        trained_params = np.zeros((len(models), self.n_folds), dtype=object)

        for i, m in enumerate(models):

            model = m['type']
            params = m['hyperparameters']

            mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

            # multiple-metric case: define a dictionary
            # scoring_dict = {'mse': mse_scorer}
            # gs = GridSearchCV(model, params, scoring=scoring_dict, refit=False, cv=self.n_folds)
            # problem: parameters will not be generated without refit! Solution: write out GridSearchCV

            if self.cross_val_strategy == 'groups':
                skf = GroupKFold(n_splits=self.n_folds)
                iterator = enumerate(skf.split(X, y, groups=g))
            else:
                # np.random.seed(0)
                skf = StratifiedKFold(n_splits=self.n_folds, shuffle=False)
                iterator = enumerate(skf.split(X, y))

            for j, (train_index, test_index) in iterator:

                X_train = X[train_index, :]
                y_train = y[train_index]
                q_train = q[train_index]
                g_train = g[train_index]
                X_test = X[test_index, :]
                y_test = y[test_index]

                skf_inner = GroupKFold(n_splits=self.n_folds_inner)
                cv_inner = skf_inner.split(X_train, y_train, groups=g_train)
                gs = GridSearchCV(model, params, scoring=mse_scorer, refit=False, return_train_score=True, cv=cv_inner)
                gs.fit(X_train, y_train, sample_weight=q_train)
                trained_params[i, j] = gs.best_params_
                model.set_params(**trained_params[i, j])
                model.fit(X_train, y_train, sample_weight=q_train)
                y_predict = model.predict(X_test)
                if indices is not None:
                    self.y_gen[indices[test_index]] = y_predict

                scores[i, j] = mean_squared_error(y_test, y_predict)
                R2[i, j] = pearsonr(y_test, y_predict)[0]
                r2[i, j] = r2_score(y_test, y_predict)
                trained_model[i, j] = model

                if isinstance(model, LinearRegression) or isinstance(model, TorchLinearRegression) \
                        or isinstance(model, TorchNN):
                    print("feature support for", model.get_params())
                    print(model.feature_support)

                # logging: GridSearchCV for (pattern p_index, model i, fold j) is complete.

                # Here's the output of CV for (pattern p_index, model i, fold j):
                # list of different hyperparameter combinations (a list of dictionaries):
                # gs.cv_results_['params']
                # training scores for different hyperparameter combinations (a list of numbers):
                # gs.cv_results_['mean_train_score']
                # validation scores for different hyperparameter combinations (a list of numbers):
                # gs.cv_results_['mean_test_score']
                # validation curve: training scores vs. hyperparameters and validation scores vs. hyperparameters
                # trained_params[i, j]    # selected hyperparameters (a dictionary)
                # scores[i, j]            # model score for the selected hyperparameters (a number)

                self.cv_mean_test_scores[p_index, i, j] = - gs.cv_results_['mean_test_score']
                self.cv_mean_train_scores[p_index, i, j] = - gs.cv_results_['mean_train_score']
                self.cv_params[p_index, i, j] = gs.cv_results_['params']

                # Also, report:
                # model type: model
                # if isinstance(model, LinearRegression) or isinstance(model, TorchLinearRegression):
                # model.feature_support   # feature support of the selected model (an array)
                # model.coef              # weights for the features indicated in model.feature_support (an array)
                # lollipop plot for strength of features selected: model.coef at model.feature_support, zero elsewhere
                # otherwise (model type not regression):
                # set model.feature_support = np.unique(self.marker_groups)
                # set model.coef = np.ones(model.feature_support.shape[0])

                if isinstance(model, LinearRegression):
                    self.feature_support_lasso[p_index, j] = model.feature_support
                    self.regression_coef[p_index, j] = model.coef

                if isinstance(model, TorchLinearRegression):
                    self.feature_support_nn_lasso[p_index, j] = model.feature_support
                    self.nn_regression_coef[p_index, j] = model.coef

                if isinstance(model, TorchNN):
                    self.feature_support_torchNN_lasso[p_index, j] = model.feature_support

        self.R2_scores[p_index] = R2
        self.r2_scores[p_index] = r2

        return scores, trained_model, trained_params

    def fit(self, X, y, Q, g=None):
        """
        Method for fitting the data to all the submodel patterns. This method first learns a completion model for
        the data. Then, it finds top (most frequent) missing value patterns in the data. For each missing value
        pattern, different models (indicated by models) are fit to the data and optimal hyper-parameters for
        each is selected via a nested cross validation.

        Args:
            X (numpy array): training data of shape (d1, d2).
            y (numpy array): training labels of shape (d1, )
            Q (numpy array): quality matrix of shape (d1, d2)
            g (numpy array): user group ids of shape (d1, )

        Returns:
            scores (dictionary): dictionary mapping pattern index to array of scores over different models and folds
        """

        markers = np.unique(self.marker_groups)
        self.default = np.mean(y)

        P = find_pattern(Q, self.n_patterns)
        self.patterns = P

        d1, d2 = np.shape(X)
        pattern_indices = np.zeros((d1,))
        self.y_gen = np.zeros((d1, ))

        for i in range(d1):
            q = Q[i, :]
            pattern_indices[i] = predict_pattern(P, q)

        print(f'Patterns found: P = {P}')

        # logging: Sub-model Patterns are found.
        # save patterns i=1,...,self.n_patterns from P[i, :] to the database; each an array of length self.n_markers.

        # imputing missing values:
        # X = np.array(X, dtype='float16')
        # X[X == 0] = np.NaN
        if self.missing_value_strategy == 'ALS':
            completion_model = ALS()
        else:
            completion_model = ImputerSKL()
        completion_model.fit(X)
        self.completion_model = completion_model

        print("Completion model found!")

        # logging: Completion model is found.
        # Nothing to report to the database.

        # overall quality indicators for each row:
        q = np.mean(Q, axis=1)

        for i in range(self.n_patterns):

            print(f'Fitting a model to pattern {i} ...')

            # find the indices of the rows of X with the same pattern as p:
            # p = P[i, :]
            # indices = fit_pattern(Q, p)
            indices = np.where(pattern_indices == i)[0]

            print(f'Number of data cases in pattern {i}: {indices.shape[0]}')

            # logging: fitting a model for pattern i
            # Report the number of data cases used for training of pattern i: len(indices)

            X1 = X[indices, :]
            y1 = y[indices]
            q1 = q[indices]
            g1 = g[indices.astype(int)]

            if self.feature_selection_strategy == "Stagewise":
                feature_support = self.backward_feature_elimination_blockwise(X1, y1, q1, g1)
            else:
                feature_support = markers

            self.feature_support[i] = feature_support

            self.scores[i], self.trained_model[i], self.trained_params[i] = \
                self.fit_submodel(X1[:, np.in1d(self.marker_groups, feature_support)], y1, indices, i, q1, g1)

        # logging: report Overall MSE/R2/r2 errors
        self.MSE = mean_squared_error(self.y_gen, y)
        self.R2 = pearsonr(self.y_gen, y)[0]
        self.r2 = r2_score(self.y_gen, y)

        print(f'MSE = {self.MSE}, R2 = {self.R2}, r2 = {self.r2}')
        print(f'y_gen = {self.y_gen[:100]}, y = {y[:100]}')

        MSE_scores = np.zeros((self.n_patterns*self.n_folds, ))
        R2_scores = np.zeros((self.n_patterns * self.n_folds,))
        r2_scores = np.zeros((self.n_patterns * self.n_folds,))

        for i in range(self.n_patterns):
            MSE_scores[i*self.n_folds:(i+1)*self.n_folds] = self.scores[i][0, :]
            R2_scores[i * self.n_folds:(i + 1) * self.n_folds] = self.R2_scores[i][0, :]
            r2_scores[i * self.n_folds:(i + 1) * self.n_folds] = self.r2_scores[i][0, :]

        # logging: Report mean and sem of MSE/R2/r2 scores over folds
        print(f'Average MSE = {np.mean(MSE_scores, axis=None)}')
        print(f'Average R2 = {np.mean(R2_scores, axis=None)}')
        print(f'Average r2 = {np.mean(r2_scores, axis=None)}')
        print(f'SEM of MSE = {sem(MSE_scores, ddof=0, axis=None)}')
        print(f'SEM of R2 = {sem(R2_scores, ddof=0, axis=None)}')
        print(f'SEM of r2 = {sem(r2_scores, ddof=0, axis=None)}')

        return self.scores, self.y_gen

    def predict_submodel(self, X, trained_model, scores):
        """
        Method for the prediction of data cases belonging to a given submodel.

        Args:
            X (numpy array): test data; each row corresponds to a data instance
            trained_model (numpy array): array of trained models over different models and folds
            scores (numpy array): array scores for the trained models over different models and folds

        Returns:
            y_predict (numpy array): test data estimates

        """

        weights = np.exp(-scores)
        weights = weights/np.sum(weights)
        y_predict = np.zeros((X.shape[0], ))

        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                y_predict += weights[i, j]*trained_model[i, j].predict(X)

        return y_predict

    def predict(self, X, Q):
        """
        Method for prediction of labels for data X and quality/missing value pattern Q. For each data case, first the
        closest missing value pattern is selected such that imputation is minimal. Once the missing value pattern is
        matched and imputation is done, prediction is performed using the learned submodel associated with that missing
        value pattern.

        Args:
            X (numpy array): data of shape (d1, d2), where d1 and d2 indicate the number of data cases and
            the number of columns/entries for each data case, respectively.
            Q (numpy array): quality indicators of shape (d1, d2)

        Returns:
            y_predict (numpy array): array of labels of shape (d1, )

        """

        d1, d2 = np.shape(X)
        markers = np.unique(self.marker_groups)
        completion_model = self.completion_model

        y_predict = np.zeros((d1, ))
        y_predict.fill(self.default)
        pattern_indices = np.zeros((d1, ))

        for i in range(d1):
            q = Q[i, :]
            if np.sum(q) != 0:
                pattern_indices[i] = predict_pattern(self.patterns, q)

        for i in range(self.n_patterns):
            indices = np.where(pattern_indices == i)[0]
            X1 = completion_model.predict(X[indices, :])

            if self.feature_support[i] is None:
                feature_support = list(markers)
            else:
                feature_support = self.feature_support[i]

            trained_model = self.trained_model[i]
            scores = self.scores[i]

            y_predict[indices] = self.predict_submodel(X1[:, np.in1d(self.marker_groups, feature_support)], trained_model, scores)

            # logging: prediction for pattern i is complete.
            # report to the databse the number of data cases predicted according to pattern i: len(indices)

        # logging: prediction is complete
        # report labels y_predict to the database

        return y_predict

    def backward_feature_elimination_blockwise(self, X, y, q, g):
        """
        This method implements the backward feature elimination algorithm.

        Args:
            X (numpy array): training data of shape (d1, d2)
            y (numpy array): training labels of shape (d1, )
            q (numpy array): quality indicators of data cases of shape (d1, )
            g (numpy array): user group ids of shape (d1, )

        Returns:
            feature_support (list): list of length n_selected_features (number of selected features)
            containing index of selected features
        """

        d1, d2 = X.shape

        if self.cross_val_strategy == 'groups':
            skf = GroupKFold(n_splits=self.n_folds)
            iterator = enumerate(skf.split(X, y, groups=g))
        else:
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True)
            iterator = enumerate(skf.split(X, y))

        markers = np.unique(self.marker_groups)

        # selected feature set, initialized to contain all features
        feature_support = list(markers)
        count = markers.shape[0]

        if not self.n_selected_features:
            # self.n_selected_features = n_markers // 2
            self.n_selected_features = markers.shape[0] - 2

        while count > self.n_selected_features:

            print(count)

            min_score = 0
            idx = None
            for i in markers:
                if i in feature_support:
                    feature_support.remove(i)
                    X_tmp = X[:, np.in1d(self.marker_groups, feature_support)]
                    score = 0
                    for train_index, test_index in iterator:
                        scores, trained_model, trained_params = self.fit_submodel(X_tmp[train_index, :], y[train_index],
                                                                         q[train_index], g[train_index])
                        y_predict = self.predict_submodel(X_tmp[test_index, :], trained_model, scores)
                        score += mean_squared_error(y[test_index], y_predict)
                    score = float(score) / self.n_folds
                    feature_support.append(i)
                    # record the feature whose absence results in the smallest error
                    if score < min_score:
                        min_score = score
                        idx = i
            # delete the feature whose absence results in the smallest error
            if idx:
                feature_support.remove(idx)
            count -= 1

        return feature_support
