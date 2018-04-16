import sys
import io
import os
from io import StringIO
import warnings
import numpy as np
import json
# import matplotlib.pyplot as plt
import mosaic_utils as mu
import datetime as dt
from features import extract_features
import cc_data_retriever as data_retriever
from random import randint
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle
import summarizer as summarizer
from minio import Minio
from minio.error import ResponseError
from pyspark import SparkContext, SparkConf
from cerebralcortex.cerebralcortex import CerebralCortex

from MLE.machine_learning_engine import MachineLearningEngine
from MLE.linear_regression import LinearRegression
from MLE.linear_regression_torch import TorchLinearRegression
from MLE.TorchNN import TorchNN
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge

import csv_utility
import userid_map

from operator import add
import socket

# ENVIRONMENT = "vm"
# ENVIRONMENT = "production"

"""
CLASS DESCRITION
"""

ENVIRONMENT = socket.gethostname()

if ENVIRONMENT == "cerebralcortex":

    print("experiment_engine: detected VM environment")

    # VM configuration
    MINIO_IP = '127.0.0.1:9000'
    MINIO_ACCESS_KEY = 'ZngmrLWgbSfZUvgocyeH'
    MINIO_SECRET_KEY = 'IwUnI5w0f5Hf1v2qVwcr'
    # STUDY_STRING = "/home/vagrant/CerebralCortex/EMS/{}.json"
    STUDY_STRING = "{}"
    cc = CerebralCortex('/home/vagrant/CerebralCortex-DockerCompose/cc_config_file/cc_vagrant_configuration.yml')
    sc = SparkContext("local", "MOSAIC")

elif ENVIRONMENT in ["dagobah10dot"]:

    print("experiment_engine: detected production environment")

    # production configuration
    cc = CerebralCortex('/cerebralcortex/code/config/cc_starwars_configuration.yml')
    minioClient = cc
    conf = SparkConf().setMaster('spark://dagobah10dot:7077').setAppName('MOSAIC').set('spark.cores.max','8').set('spark.ui.port','4099').setExecutorEnv('PYTHONPATH',str(os.getcwd()))
    sc = SparkContext(conf=conf)
    sc.addPyFile('/cerebralcortex/code/CerebralCortex/dist/MD2K_Cerebral_Cortex-2.2.2-py3.6.egg')

    MINIO_IP = cc.config['minio']['host'] + ':' + str(cc.config['minio']['port'])
    MINIO_ACCESS_KEY = cc.config['minio']['access_key']
    MINIO_SECRET_KEY = cc.config['minio']['secret_key']
    STUDY_STRING = "/cerebralcortex/code/ems/EMS/{}"

    print(MINIO_IP,MINIO_ACCESS_KEY,MINIO_SECRET_KEY)
    
else:
    print("unknown environment!")


def main():
    """
    The experiment engine runs the whole EMS: it is responsible for reading in the experiment design document
    (EDD) that describes the parameters of an experiment, initiating summarization of the specified data,
    then calling on the machine learning engine code to learn a model and make predictions for the given
    data set.
    """

    if not len(sys.argv) == 3:
        print("required arguments: [edd name] [run type: fit/predict/fit-predict]")
        return

    # filename used both to open the EDD and to save/load models
    filename = sys.argv[1]
    run_type = sys.argv[2]

    # retrieve experiment parameters from JSON doc
    experiment_file = open(STUDY_STRING.format(filename))
    experiment = json.load(experiment_file)

    # print(experiment_file)

    experiment_type, streams, compound_streams, target, models, loss_function, start_time, end_time, wb_list, prediction_subjects, prediction_start, prediction_end = mu.get_parameters(experiment)
    
    # only arrays of whitelisted IDs supported right now (files and blacklist coming soon): fatal
    if "whitelist" in wb_list:
        if wb_list["type"] == "list":
            subjects = wb_list["whitelist"]
        else:
            print("files not supported yet")
            return
    else:
        print("blacklist not supported yet")
        return

    # preload all the necessary summaries (uses Spark on the other side)
    # job_type = "by-subject"
    job_type = "none"
    summarizer.compute_summaries(sc, filename, job_type)

    if run_type == "fit" or run_type == "fit-predict":

        print("starting learning job")

        # learn_model(filename)

        learning_job = sc.parallelize([filename])
        job = learning_job.map(learn_model)

        job_results = job.reduce(add)
        print("results: " + str(job_results))

    if run_type == "predict" or run_type == "fit-predict":

        print("starting prediction job")

        predict(filename)

        #pred_rdd = sc.parallelize([filename])
        #pred_job = pred_rdd.map(predict)

        #pred_results = pred_job.reduce(add)
        # print("results: " + str(pred_results))


def learn_model(edd):
    """
    Primary model-learning function, run as a Spark job.

    Args:
        edd (str): Path to the EDD for the experiment

    Returns:
        unnamed val (int): Value reporting the success or failure of the learning job
    """

    minioClient = Minio(MINIO_IP,
                  access_key=MINIO_ACCESS_KEY,
                  secret_key=MINIO_SECRET_KEY,
                  secure=False)

    # get experiment parameters
    experiment_file = open(STUDY_STRING.format(edd))
    experiment = json.load(experiment_file)
    import mosaic_utils as mu
    experiment_type, streams, compound_streams, target, models, loss_function, start_time, end_time, wb_list, prediction_subjects, prediction_start, prediction_end = mu.get_parameters(experiment)

    # only arrays of whitelisted IDs supported right now (files and blacklist coming soon): fatal
    if "whitelist" in wb_list:
        if wb_list["type"] == "list":
            subjects = wb_list["whitelist"]
        else:
            print("files not supported yet")
            return
    else:
        print("blacklist not supported yet")
        return

    # get previously-summarized data
    X1, y1, Q1, g1, t1 = summarizer.get_summary_table(subjects, streams, compound_streams, target, start_time, end_time, "histogram", experiment_type)

    # mapping userids to UMN's 10-character format
    g1 = userid_map.perform_map(g1)

    y1 = np.array(y1, dtype=float)
    indices1 = np.isfinite(y1)
    indices2 = np.sum(Q1, axis=1) != 0
    indices = np.logical_and(indices1, indices2)

    date = X1[:, 0].copy()
    X = np.array(X1[indices, :][:, 1:], dtype=float)
    y = y1[indices].copy()
    Q = Q1[indices, :].copy()
    g = g1[indices].copy()

    np.set_printoptions(threshold=np.nan)
    # print(f'Original Q = {Q}')

    # defining marker groups in the format required by MLE.
    n_bins = 10
    n_markers = Q.shape[1]
    marker_groups = []
    for i in range(n_markers):
        marker_groups += [i] * n_bins
    markers, repeats = np.unique(marker_groups, return_counts=True)
    Q = np.repeat(Q, repeats, axis=1)

    # print("quality matrix:\n{}".format(str(Q)))
    print("X: {}".format(X))

    print("X.shape: {}\ny.shape: {}\nQ.shape: {}\ng.shape: {}".format(X.shape, y.shape, Q.shape, g.shape))

    print("X: {}\ny: {}\nQ: {}\ng: {}".format(X, y, Q, g))

    print("indices: {}, indices.sum(): {}".format(list(indices), indices.sum()))
    if indices.sum() == 0:
        print("Aborting: No data available for learning job!")
        return 0
    # if np.unique(g).size < 9:
    #     print("Aborting: Not enough users available for learning job!")
    #     return 0


    # update_status("Loading synthetic data")
    # loading the synthetic data:
    # (X, y, Q, g, marker_groups) = np.load('MLE/synthetic_data/test_data.npy')
    # update_status("Synthetic data loaded")

    import dev
    target1 = dev.get_label_field_name(target['name'])
    print(f'target1 = {target1}')

    model_map = {'DecisionTreeRegressor': DecisionTreeRegressor(criterion=loss_function),
                 'RandomForestRegressor': RandomForestRegressor(criterion=loss_function),
                 'LinearRegression': LinearRegression(marker_groups=marker_groups),
                 'TorchLinearRegression': TorchLinearRegression(marker_groups=marker_groups),
                 'TorchNN': TorchNN(marker_groups=marker_groups, optim='l-bfgs', layers=[None, 8, 4]),
                 'SVR': SVR(),
                 'Ridge': Ridge(random_state=0)}

    model = models[0]
    model["type"] = model_map[model["type"]]

    if experiment_type == 'daily':
        if target1 == 'tob.quantity.d':
            # MLE object:
            mle = MachineLearningEngine(models=[model], marker_groups=marker_groups, n_folds=3, n_folds_inner=2,
                                        loss_function=loss_function, n_patterns=2)
        else:
            # MLE object:
            mle = MachineLearningEngine(models=[model], marker_groups=marker_groups, n_folds=3, n_folds_inner=3,
                                        loss_function=loss_function, n_patterns=2)

    elif experiment_type == 'intake':
        # MLE object:
        mle = MachineLearningEngine(models=[model], marker_groups=marker_groups, n_folds=2, n_folds_inner=2,
                                    loss_function=loss_function, n_patterns=1)
    else:
        print("Aborting: Experiment type is not valid!")
        return 0

    # save data for reproducing the segfault:
    # np.save('sfdata.npy', (X, y, Q, g, marker_groups))

    # fitting the model:
    scores, y_gen = mle.fit(X, y, Q, g)

    # print("learned models with average score " + str(np.mean(scores)))
    print(f'CV scores = {scores}')
    print(f'y generalization = {y_gen}')

    # adding data cases with no labels back in:
    y1_gen = np.full(y1.shape, np.NaN)
    y1_gen[indices] = np.array(y_gen)

    time_zone = np.repeat('', len(y1_gen))

    add_name = 'Results/' + experiment_type + '_fit_'
    if experiment_type == 'daily':
        csv_utility.write_csv_daily(add_name, g1, date, time_zone, target1, y1_gen)
    elif experiment_type == 'intake':
        csv_utility.write_csv_initial(add_name, g1, date, time_zone, target1, y1_gen)
    else:
        print("Aborting: Experiment type is not valid!")

    # TODO: save models in loop
    m = io.BytesIO(pickle.dumps(mle))
    # m = io.BytesIO(pickle.dumps(models))
    # m = io.BytesIO(pickle.dumps(clf.best_params_))

    experiment_bucket_name = os.path.basename(edd)

    if not minioClient.bucket_exists(experiment_bucket_name):
        minioClient.make_bucket(experiment_bucket_name)

    try:
        model_name = dt.datetime.now().strftime("%Y-%m-%d-%H-%M")
        print("model saved to {} with id {}".format(experiment_bucket_name, str(minioClient.put_object(experiment_bucket_name, model_name, m, len(m.getvalue())))))
    except ResponseError as err:
        print(err)

    return 1


def predict(edd):
    """
    Launches a separate Spark job to do prediction once a model has been learned.

    Args:
        edd (str): Path to the EDD for the experiment

    Returns:
        unnamed val (int): Value reporting the success or failure of the learning job
    """

    minioClient = Minio(MINIO_IP,
                  access_key=MINIO_ACCESS_KEY,
                  secret_key=MINIO_SECRET_KEY,
                  secure=False)

    experiment_file = open(STUDY_STRING.format(edd))
    experiment = json.load(experiment_file)
    experiment_bucket_name = sys.path.basename(edd)
    experiment_type, streams, compound_streams, target, models, loss_function, start_time, end_time, wb_list, prediction_subjects, prediction_start, prediction_end = mu.get_parameters(experiment)

    # prediction

    if len(prediction_subjects) > 0:
        X1, y1, Q1, g1, t1 = summarizer.get_summary_table(prediction_subjects, streams, compound_streams, target, prediction_start, prediction_end, "histogram", experiment_type)

        # mapping userids to UMN's 10-character format:
        g1 = userid_map.perform_map(g1)

        y1 = np.array(y1, dtype=float)
        indices1 = np.isfinite(y1)
        indices2 = np.sum(Q1, axis=1) != 0
        indices = np.logical_and(indices1, indices2)

        date = X1[:, 0].copy()
        X = np.array(X1[indices, :][:, 1:], dtype=float)
        y = y1[indices].copy()
        Q = Q1[indices, :].copy()
        g = g1[indices].copy()

        np.set_printoptions(threshold=np.nan)
        print(f'Original Q = {Q}')

        marker_groups = [0] * 10 + [1] * 10
        markers, repeats = np.unique(marker_groups, return_counts=True)
        Q = np.repeat(Q, repeats, axis=1)

        # print("quality matrix:\n{}".format(str(Q)))
        # print("X: {}".format(X))

        print("X.shape: {}\ny.shape: {}\nQ.shape: {}\ng.shape: {}".format(X.shape, y.shape, Q.shape, g.shape))

        print("X: {}\ny: {}\nQ: {}\ng: {}".format(X, y, Q, g))
        
        for saved_model in minioClient.list_objects(experiment_bucket_name):
            print("found model {}".format(saved_model.object_name.encode('utf-8')))
            
            data = minioClient.get_object(experiment_bucket_name, saved_model.object_name.encode('utf-8'))
            with io.BytesIO() as file_data:
                for d in data.stream(32*1024):
                    file_data.write(d)

                file_data.seek(0)

                mle = pickle.load(file_data)

                # loading the synthetic data:
                # (X, y, Q, g, marker_groups) = np.load('MLE/synthetic_data/test_data.npy')

                # prediction:
                y_pred = mle.predict(X, Q)

                print(f'True labels = {y}')
                print(f'Regression Estimates = {y_pred}')

                # adding data cases with no labels back in:
                y1_pred = np.full(y1.shape, np.NaN)
                y1_pred[indices] = np.array(y_pred)

                time_zone = np.repeat('', len(y1_pred))

                import dev
                target1 = dev.get_label_field_name(target["name"])

                add_name = experiment_type + '_predict_'
                if experiment_type == 'daily':
                    csv_utility.write_csv_daily(add_name, g1, date, time_zone, target1, y1_pred)
                elif experiment_type == 'intake':
                    csv_utility.write_csv_initial(add_name, g1, date, time_zone, target1, y1_pred)
                else:
                    print("Aborting: Experiment type is not valid!")

    else: 
        print("nothing to predict")
        return 0

    return 1

    # plt.figure()
    # formats = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko', 'wo', 'bx', 'gx', 'rx', 'cx', 'mx', 'kx', 'bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko', 'wo', 'bx', 'gx', 'rx', 'cx']
    # for i in range(0,len(y),10): # only plot 1/10th of the points, it's a lot of data!
    #     plt.plot(X[i,0], X[i, 1], formats[int(y[i])])

    # plt.show()

if __name__ == '__main__':
  main()
