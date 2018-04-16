
import numpy as np
from scipy.stats import sem


def mle_output_summarizer(mle):
    """This module summarizes the output of an instance of the machine_learning_engine."""

    # MSE (Mean Squared Error):
    MLE = mle.MSE
    # R2 (Pearson Coefficient):
    R2 = mle.R2
    # r2 (Coefficient of Determination):
    r2 = mle.r2

    MSE_scores = np.zeros((mle.n_patterns * mle.n_folds,))
    R2_scores = np.zeros((mle.n_patterns * mle.n_folds,))
    r2_scores = np.zeros((mle.n_patterns * mle.n_folds,))

    for p_index in range(mle.n_patterns):

        # missing data pattern:
        p = mle.P[p_index, :]

        # feature support for the pattern: (useful only when stagewise feature selection is used.)
        feature_support = mle.feature_support[p_index]

        # test scores for different models: Don't report this for now.
        # test_scores = np.mean(mle.scores[p_index], axis=0)

        MSE_scores[p_index * mle.n_folds:(p_index + 1) * mle.n_folds] = mle.scores[i][0, :]
        R2_scores[p_index * mle.n_folds:(p_index + 1) * mle.n_folds] = mle.R2_scores[i][0, :]
        r2_scores[p_index * mle.n_folds:(p_index + 1) * mle.n_folds] = mle.r2_scores[i][0, :]

        for i, m in enumerate(mle.models):

            # model type
            model = m['type']

            # test scores for the i-th model (model 'model'):
            test_score = test_scores[i]

            for j in range(mle.n_folds):

                # hyperparameters: list of different hyperparameter combinations (a list of dictionaries):
                params = mle.cv_params[p_index, i, j]

                # training scores: training scores for different hyperparameter combinations (a list of numbers):
                mean_train_scores = mle.cv_mean_train_scores[p_index, i, j]

                # validation scores: validation scores for different hyperparameter combinations (a list of numbers):
                mean_test_scores = mle.cv_mean_test_scores[p_index, i, j]

                # plot validation curve: training scores vs. hyperparameters and validation scores vs. hyperparameters

                # selected hyperparameters (for fold j):
                selected_params = mle.trained_params[p_index][i, j]

                if m['type'] == "LinearRegression":
                    feature_support_lasso = mle.feature_support_lasso[p_index, j]
                    lasso_coef = mle.regression_coef[p_index, j]
                    # plot lasso_coef at feature_support_lasso, zeros elsewhere in list(range(n_markers))

                if m['type'] == "TorchLinearRegression":
                    feature_support_nn_lasso = mle.feature_support_nn_lasso[p_index, j]
                    nn_lasso_coef = mle.nn_regression_coef[p_index, j]
                    # plot nn_lasso_coef at feature_support_nn_lasso, zeros elsewhere in list(range(n_markers))

                if m['type'] == "TorchNN":
                    feature_support_torchNN_lasso = mle.feature_support_torchNN_lasso[p_index, j]
                    # report feature support for TorchNN

    # Average MSE Across Folds/Models:
    average_MSE = np.mean(MSE_scores)
    # Average R2 Across Folds/Models:
    average_R2 = np.mean(R2_scores)
    # Average r2 Across Folds/Models:
    average_r2 = np.mean(r2_scores)
    # SEM (Standard error of the Mean) of MSE Across Folds/Models:
    sem_MSE = sem(MSE_scores, ddof=0)
    # SEM of R2 Across Folds/Models:
    sem_R2 = sem(R2_scores, ddof=0)
    # SEM of r2 Across Folds/Models:
    sem_r2 = sem(r2_scores, ddof=0)

    return True
