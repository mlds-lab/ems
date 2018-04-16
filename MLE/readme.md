

## Machine Learning Engine

The core of the machine learning algorithms is implemented in the MachineLearningEngine class. This class implements the Sci-Kit-Learn fit/predict interface.

### Usage:

Inputs:

- `models`: Classification or Regression models defined as a list of dictionaries (each dictionary corresponds to a model). Format of each dictionary: `{“type”: ClassiferOrRegressorObject, “hyperparameters”: {“alpha”: [1,2,3], “beta”: [1,2,3]} }`, where “hyperparameters” are mapped to a dictionary of different hyperparameters mapped to lists of possible values for hyperparameters. Note that hyperparameters (i.e. their number and their values) are dictated by the classifier or regressor class.
- `X`: data of size `(d1, d2)`, where d1 and d2 indicate the number of data cases and the number of columns/entries for each data case, respectively.
- `y`: labels of size `(d1, )`
- `Q`: data quality indicator array of size `(d1, d2)`, where for all i and j, `0 <= Q[i, j] <= 1` and indicates how much confidence we have in the value of `X[i, j]`. `Q[i, j] = 0` implies that `X[i, j]` is missing and `Q[i, j] = 1` implies that we are very confident in the measurement `X[i, j]`. 
- `g`: user ids of size `(d1, )`. If two data cases have the same user id, they belong to the same user.
- `marker_groups`: marker group labels of size `(d2, )`; example: if we have three markers each of sizes `2,3,4` respectively, marker_groups becomes `[0,0,1,1,1,2,2,2,2]`.
- `n_patterns`: Number of top (most frequent) missing value patterns in the data.

Methods:

- `fit`: This method first learns a completion/imputation model for the data. Then, it finds top (most frequent) missing value patterns in the data. For each missing value pattern, different models (indicated by “models”) are fit to the data and optimal hyper-parameters for each model is selected via a nested cross validation. 

- `predict`: Method for prediction of labels for data X and quality/missing value pattern Q. For each data case, first the closest missing value pattern is selected such that imputation is minimal. Once the missing value pattern is matched and imputation is done, prediction is performed using the learned submodel associated with that missing value pattern.


### Example Usage:
```
import numpy as np
from machine_learning_engine import MachineLearningEngine
from linear_regression import LinearRegression

“””loading the sample data:”””
(X, y, Q, g, marker_groups) = np.load('synthetic_data/test_data.npy')

“””defining the learning model:”””
models = [{"type": LinearRegression(marker_groups=marker_groups), 
	    "hyperparameters": {"alpha": [1, 0.1, 0.01, 0.001], "lbda": [6, 7, 8, 9, 10]}}]

“””creating an object of the MachineLearningEngine:”””
mle = MachineLearningEngine(models=models, marker_groups=marker_groups, n_patterns=5)

“””fitting the models:”””
mle.fit(X, y, Q, g)

“””making predictions:”””
y_pred = mle.predict(X, Q)
```