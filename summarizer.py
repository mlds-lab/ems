import sys
import os
import io
import math
from io import StringIO
from io import BytesIO
import warnings
import numpy as np
import json
# import matplotlib.pyplot as plt
import mosaic_utils as mu
import datetime as dt
import time
#from features import extract_features
import cc_data_retriever as data_retriever
from random import randint
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle
from minio import Minio
from minio.error import ResponseError
from cerebralcortex.cerebralcortex import CerebralCortex
from pyspark import SparkContext
from operator import add
import dev as dev
import socket

# ENVIRONMENT = "vm"
# ENVIRONMENT = "production"

ENVIRONMENT = socket.gethostname()
print(ENVIRONMENT)

if ENVIRONMENT == "cerebralcortex":

    print("summarizer: detected VM environment")

    # VM Configuration:
    MINIO_IP = '127.0.0.1:9000'
    MINIO_ACCESS_KEY = 'ZngmrLWgbSfZUvgocyeH'
    MINIO_SECRET_KEY = 'IwUnI5w0f5Hf1v2qVwcr'
    cc = CerebralCortex('/home/vagrant/CerebralCortex-DockerCompose/cc_config_file/cc_vagrant_configuration.yml')
    minioClient = cc
    # STUDY_STRING = "/home/vagrant/CerebralCortex/EMS/{}.json"
    STUDY_STRING = "{}"

elif '10dot' in ENVIRONMENT or 'memphis' in ENVIRONMENT:

    print("summarizer: detected production environment")

    # production configuration:
    cc = CerebralCortex('/cerebralcortex/code/config/cc_starwars_configuration.yml')
    minioClient = cc
    MINIO_IP = cc.config['minio']['host'] + ':' + str(cc.config['minio']['port'])
    MINIO_ACCESS_KEY = cc.config['minio']['access_key']
    MINIO_SECRET_KEY = cc.config['minio']['secret_key']
    STUDY_STRING = "/cerebralcortex/code/ems/EMS/{}"

else:
    print("unknown environment!")


mC = Minio(MINIO_IP,
                  access_key=MINIO_ACCESS_KEY,
                  secret_key=MINIO_SECRET_KEY,
                  secure=False)

def get_summary_table(subjects, streams, compound_streams, target, start, end, summ_function, experiment_type):
    """
    Retrieves a summary table of the specified summarization type.

    Args:
        subjects (List(str)): uuids of subjects to include in summary table
        streams (List(str)): Names of marker streams to include in summary table
        compound_streams (List(str)): Names of compound streams to include in summary table
        target (str): Label to filter against
        start (str): Start of data range to include in summary table
        end (str): End of data range to include in summary table
        summ_function (str): Summarization function
        experiment_type (str): Experiment type: "daily" or "intake" (for IGTB)

    Returns:
        X (numpy array): A matrix of histograms
        y (numpy array): A list of truth labels
        Q (numpy array): A matrix of quality scores
        g (numpy array): A list of participants for each row in X
        t (numpy array): A list of timezones for each row in X.
    """
    if summ_function == 'histogram':

        X, y, Q, g, t = get_histogram_table(subjects, streams, compound_streams, target, start, end, experiment_type)

        return X, y, Q, g, t

    # TODO: additional summarization function types will go here...
    else:
        
        print(summ_function + " not supported yet, defaulting to histogram")

        X, y, Q, g, t = get_histogram_table(subjects, streams, compound_streams, target, start, end, experiment_type)

        return X, y, Q, g, t

def retrieve_histogram(sub, obj_name):
    """
    Retrieves a summary table of the specified summarization type.

    Args:
        sub (str): uuid of subject
        obj_name (str): Name of histogram file to retrieve

    Returns:
        date_hist (numpy array): Reconstructed histogram from stored file.
    """
    if bucket_contains_object(sub, obj_name):

        print("found file " + obj_name + " in bucket " + sub)

        date_hist = get_np_array_from_minio(sub, obj_name)
        date_hist = np.reshape(date_hist, (1, len(date_hist)))
        return date_hist

    else:
        print("didn't find file " + obj_name + " in bucket " + sub)
        return None

def get_histogram_table(subjects, streams, compound_streams, target, start, end, experiment_type, bins=10):
    """
    Retrieves a histogram table of histograms for learning or prediction.

    For all subjects, loop through all streams, gather histograms for all available user dates.  Repeat
    for compound streams.

    Args:
        subjects (List(str)): uuids of subjects to include in summary table
        streams (List(str)): Names of marker streams to include in summary table
        compound_streams (List(str)): Names of compound streams to include in summary table
        target (str): Label to filter against
        start (str): Start of data range to include in summary table
        end (str): End of data range to include in summary table
        summ_function (str): Summarization function
        experiment_type (str): Experiment type: "daily" or "intake" (for IGTB)

    Returns:
        X (numpy array): A matrix of histograms
        y (numpy array): A list of truth labels
        Q (numpy array): A matrix of quality scores
        g (numpy array): A list of participants for each row in X
        t (numpy array): A list of timezones for each row in X.
    """
    # preload summaries
    all_summaries = np.zeros((0, 1 + (len(streams) + len(compound_streams)) * bins)) # date column + bins * marker streams
    quality_matrix = np.zeros((0, len(streams) + len(compound_streams))) # one cell for each summary -> len(dates) rows x len(streams) cols
    group_list = []
    target_list = []
    tz_list = []
    tz = None

    # target_sample_type = target["sample-type"]
    target_sample_type = "continuous"

    for sub in subjects:

        subject_streams = cc.get_user_streams(sub)

        available_dates = data_retriever.available_dates_for_user(cc, sub)

        print("get_histogram_table(): found {} available dates for user {}: {}".format(len(available_dates), sub, list(available_dates)))

        if experiment_type == "daily":
            available_dates_col = np.reshape(available_dates, (len(available_dates), 1))

            subject_hist = np.zeros((len(available_dates), 0))
            subject_hist = np.append(subject_hist, available_dates_col, axis=1)

            subject_quality = np.zeros((len(available_dates), 0))

        elif experiment_type == "intake":
            subject_hist = np.zeros((1, 0))
            subject_hist = np.append(subject_hist, [[0]], axis=1)
            subject_quality = np.zeros((1, 0))

        for s in streams:

            stream_with_field = s["name"]

            stream, field = mu.stream_and_field(stream_with_field)

            stream_hist = np.zeros((0, bins))
            stream_quality = np.asarray([])

            if experiment_type == "daily":

                for date in available_dates:
                    date_hist = np.asarray([])

                    obj_name = "{}-{}-{}-{}".format("histogram", bins, stream, date)
                    date_hist = retrieve_histogram(sub, obj_name)

                    stream_hist = np.append(stream_hist, date_hist[:, 2:], axis=0) # accounts for prepended validity score
                    stream_quality = np.append(stream_quality, date_hist[:, 1:2])

                subject_hist = np.append(subject_hist, stream_hist, axis=1)
                stream_quality = np.reshape(stream_quality, (len(stream_quality), 1))
                subject_quality = np.append(subject_quality, stream_quality, axis=1)

                print("subject_hist: {}, all_summaries: {}".format(str(subject_hist.shape), str(all_summaries.shape)))

            elif experiment_type == "intake":

                obj_name = "{}-{}-{}-{}".format("histogram", bins, stream, "intake")

                date_hist = retrieve_histogram(sub, obj_name)

                stream_hist = np.append(stream_hist, date_hist[:, 2:], axis=0) # accounts for prepended validity score
                stream_quality = np.append(stream_quality, date_hist[:, 1:2])

                subject_hist = np.append(subject_hist, stream_hist, axis=1)
                stream_quality = np.reshape(stream_quality, (len(stream_quality), 1))
                subject_quality = np.append(subject_quality, stream_quality, axis=1)

            else:
                print("unrecognized experiment type: {}".format(experiment_type))

        for streams in compound_streams:
            stream_hist = np.zeros((0, bins))
            stream_quality = np.asarray([])
            
            if streams["function"] == "filter":
                compound_params = streams["params"]
                target_stream = compound_params["target-stream"]["name"]
                filter_stream = compound_params["filter-stream"]["name"]
                filter_value = compound_params["threshold-value"]
                threshold_type = compound_params["threshold-type"]

            else:
                print("filter type not recognized")
                break

            if experiment_type == "daily":
                for date in available_dates:
                    date_hist = np.asarray([])

                    # filename to store
                    compound_stream_name = "{}-{}-{}-{}-{}".format(target_stream, filter_stream, "filter", threshold_type, filter_value)
                    obj_name = "{}-{}-{}-{}".format("histogram", bins, compound_stream_name, date)

                    date_hist = retrieve_histogram(sub, obj_name)
                    stream_hist = np.append(stream_hist, date_hist[:, 2:], axis=0) # accounts for prepended validity score
                    stream_quality = np.append(stream_quality, date_hist[:, 1:2])

                subject_hist = np.append(subject_hist, stream_hist, axis=1)
                stream_quality = np.reshape(stream_quality, (len(stream_quality), 1))
                subject_quality = np.append(subject_quality, stream_quality, axis=1)
                # print(str(quality_matrix))

                print("subject_hist: {}, all_summaries: {}".format(str(subject_hist.shape), str(all_summaries.shape)))
            
            elif experiment_type == "intake":
                date_hist = np.asarray([])

                # filename to store
                compound_stream_name = "{}-{}-{}-{}-{}".format(target_stream, filter_stream, "filter", threshold_type, filter_value)
                obj_name = "{}-{}-{}-{}".format("histogram", bins, compound_stream_name, "intake")

                date_hist = retrieve_histogram(sub, obj_name)
                stream_hist = np.append(stream_hist, date_hist[:, 2:], axis=0) # accounts for prepended validity score
                stream_quality = np.append(stream_quality, date_hist[:, 1:2])

                subject_hist = np.append(subject_hist, stream_hist, axis=1)
                stream_quality = np.reshape(stream_quality, (len(stream_quality), 1))
                subject_quality = np.append(subject_quality, stream_quality, axis=1)

            else:
                print("unknown experiment type: {}".format(experiment_type))

        print("completed subject {}, len(target_list): {}".format(sub, len(target_list)))

        # append each subject's table onto all_summaries, axis=0

        print("all_summaries.shape: {}, subject_hist.shape: {}".format(all_summaries.shape, subject_hist.shape))
        
        all_summaries = np.append(all_summaries, subject_hist, axis=0)
        quality_matrix = np.append(quality_matrix, subject_quality, axis=0)

        if experiment_type == "daily":
            for date in available_dates:

                if target["name"] not in subject_streams:
                    day_target_data = np.nan
                else:
                    day_target_data = target_data_for_day(sub, target["name"], date, target_sample_type)

                print("day target data: {}".format(day_target_data))

                target_list.append(day_target_data)

                # piggyback subject IDs and timezone
                group_list.append(sub)
                tz_list.append(tz)

        elif experiment_type == "intake":
            target_list.append(initial_target_data(sub, target["name"]))
            group_list.append(sub)
            tz_list.append(tz)

            print("single stream intake len(target_list): {} for subject {}".format(len(target_list), sub))

        else:
            print("unrecognized experiment type: {}".format(experiment_type))

        target_array = np.asarray(target_list)
        group_matrix = np.asarray(group_list)
        tz_matrix = np.asarray(tz_list)

        print("returning X: {}, y: {}, Q: {}, g: {}".format(list(all_summaries), list(target_array), list(quality_matrix), list(group_matrix)))

    return all_summaries, target_array, quality_matrix, group_matrix, tz_matrix

def target_data_for_day(subject, target, day, sample_type):
    """
    Retrieves target label for a particular subject and day.

    Args:
        subject (str): uuid of subject
        target (str): Name of target stream
        day (str): Formatted date to to query for target value
        sample_type (str): Continuous or discrete

    Returns:
        label_point (float): Label value if available, otherwise nan.
    """
    label_points = []

    stream_ids = cc.get_stream_id(subject, target)

    for id in stream_ids:
        stream_uuid = id["identifier"]
        label_data = cc.get_stream(stream_uuid, subject, day).data
        if not label_data == None:
            label_points.extend(label_data)

    if len(label_points) >= 1:

        # get the only label from the data set, get the data point's sample, then get the value from the list
        label_point = label_points[0].sample[0]

        print("label point: {}".format(label_point))

        if math.isnan(label_point):
            return np.nan

        else:    
            return label_point

    else:
        print("warning: found {} label points for stream/user/day: {}/{}/{}:\n{}".format(len(label_points), target, subject, day, list(label_points)))
        return np.nan

def initial_target_data(subject, target):
    """
    Retrieves initial target data.

    Args:
        subject (str): uuid of subject
        target (str): Name of target stream to retrieve

    Returns:
        label_val (float): Label value if available, otherwise nan.
    """

    stream_ids = cc.get_stream_id(subject, target)
    label_points = []

    for id in stream_ids:
        stream_uuid = id["identifier"]

        for day in data_retriever.available_dates_for_stream(cc, stream_uuid):
            label_data = cc.get_stream(stream_uuid, subject, day).data

            if not label_data == None:
                label_points.extend(label_data)

    if len(label_points) >= 1:
        label_point = label_points[0]

        # print(str(label_point.sample))

        label_val = label_point.sample[0]

        if not label_val * 0 == 0:
            return np.nan

        else:    
            return label_val

    else:
        print("warning: found {} label points for stream/user: {}/{}".format(len(label_points), target, subject))
        return np.nan

def compute_summaries(sc, edd, parallelism="all"):
    """
    Builds a set of dictionaries that get serialized and passed to Spark jobs to summarize data for
    and experiment.

    Available parallelization options are by-subject, by-stream, by-date, or all.

    Args:
        sc (SparkContext): The SparkContext object to be used to run summarization jobs
        edd (str): Path of EDD file to open and read
        parallelism (str): one of the available parallelization schemes
    """
    if parallelism not in ["all", "by-subject", "by-stream", "by-date"]:
        print("'{}' is not a recognized job type: defaulting to 'all'".format(parallelism))
        parallelism = "all"

    json_string = open(STUDY_STRING.format(edd))
    experiment = json.load(json_string)
    experiment_type, streams, compound_streams, target, models, loss_function, start_time, end_time, wb_list, prediction_subjects, prediction_start, prediction_end = mu.get_parameters(experiment)
    if "whitelist" in wb_list:
            if wb_list["type"] == "list":
                subjects = wb_list["whitelist"]
            else:
                print("files not supported yet")
                return
    else:
        print("blacklist not supported yet")
        return

    # FIXME: if prediction has different start and end dates, it should be summarized independently
    subjects = subjects + prediction_subjects

    print("added {} to list of subjects for summary computation".format(list(prediction_subjects)))

    if parallelism == "by-subject":

        print("parallelizing by subject")

        # parallelizable list of spark jobs (one per subject)
        job_list = []

        # build up dictionary, write to string, pass to write_..._for_subs...()
        for i in range(0, len(subjects)):
            job_dict = {}
            job_dict["subjects"] = [subjects[i]]
            job_dict["streams"] = streams
            job_dict["target"] = target
            job_dict["start-time"] = start_time
            job_dict["end-time"] = end_time
            job_dict["experiment-type"] = experiment_type

            if len(compound_streams) > 0:
                job_dict["compound-streams"] = compound_streams

            job_list.append(json.dumps(job_dict))
        
        print("generating rdd...")

        summ_rdd = sc.parallelize(job_list)

        job = summ_rdd.map(write_summaries_for_subs_streams_target_start_end)
        res = job.reduce(add)

        print("map/reduce complete")

    if parallelism == "by-stream":

        print("parallelizing by stream")

        # parallelizable list of spark jobs (one per stream)
        job_list = []

        # build up dictionary, write to string, pass to write_..._for_subs...()
        for i in range(0, len(streams)):
            job_dict = {}
            job_dict["subjects"] = subjects
            job_dict["streams"] = [streams[i]]
            job_dict["target"] = target
            job_dict["start-time"] = start_time
            job_dict["end-time"] = end_time
            job_dict["experiment-type"] = experiment_type

            if len(compound_streams) > 0:
                job_dict["compound-streams"] = compound_streams

            job_list.append(json.dumps(job_dict))
        
        print("generating rdd...")

        summ_rdd = sc.parallelize(job_list)

        job = summ_rdd.map(write_summaries_for_subs_streams_target_start_end)
        res = job.reduce(add)

        print("map/reduce complete")

    elif parallelism == "by-date":

        print("parallelizing by date")

        available_dates = get_available_dates(subjects, [s["name"] for s in streams])

        # parallelizable list of spark jobs (one per date)
        job_list = []

        for i in range(0, len(available_dates)):
            job_dict = {}
            job_dict["subjects"] = subjects
            job_dict["streams"] = streams
            job_dict["target"] = target
            job_dict["start-time"] = start_time
            job_dict["end-time"] = end_time
            job_dict["experiment-type"] = experiment_type

            if len(compound_streams) > 0:
                job_dict["compound-streams"] = compound_streams

            job_list.append(json.dumps(job_dict))
        
        print("generating rdd...")

        summ_rdd = sc.parallelize(job_list)

        job = summ_rdd.map(write_summaries_for_subs_streams_target_start_end)
        res = job.reduce(add)

        print("map/reduce complete")

    elif parallelism == "all":

        print("running summarization as a single batch job")

        # parallelizable list of spark jobs (only one in this case)
        job_list = []

        # build up dictionary, write to string, pass to write_..._for_subs...()
        job_dict = {}
        job_dict["subjects"] = subjects
        job_dict["streams"] = streams
        job_dict["target"] = target
        job_dict["start-time"] = start_time
        job_dict["end-time"] = end_time
        job_dict["experiment-type"] = experiment_type

        if len(compound_streams) > 0:
                job_dict["compound-streams"] = compound_streams

        job_list.append(json.dumps(job_dict))
        
        summ_rdd = sc.parallelize(job_list)
        job = summ_rdd.map(write_summaries_for_subs_streams_target_start_end)
        res = job.reduce(add)

def write_daily_stream_histograms(sub, stream, stream_id, field, user_dates, label_name, experiment_type, period_days=1, period_hours=0, bins=10):
    
    """
    Create a histogram representing period_days' worth of data.  Write the results to minio. 
    
    Any date for which no label is available is irrelevant, so an empty histogram is written.

    Args:
        sub (str): uuid of subject
        stream (str): The stream to summarize
        stream_id (str): uuid of stream to summarize
        field (str): The field within a compound sample to retrieve
        user_dates (List(str)): String representations of all available data-collection dates for the user
        label_name (str): Name of the target stream
        experiment_type (str): Daily vs intake
        period_days (int): Number of days to summarize for each label
        bins (int): Number of bins to histogram data into
    """

    # check for availability here, not in data retriever
    #FIXME: this should be done in the method that calls this method, then passed in
    user_streams = cc.get_user_streams(sub)

    if not label_name in user_streams:
        print("no data found for target {} for user {}: writing empty histograms".format(label_name, sub))

        for date in user_dates:
            obj_name = "{}-{}-{}-{}".format("histogram", bins, stream, str(int(date)))
            write_empty_histogram_for_date(sub, stream, date, experiment_type, bins=10)

        return

    for date in user_dates:

        # filename to store
        obj_name = "{}-{}-{}-{}".format("histogram", bins, stream, date)

        # TODO: forced summarization should be disabled once file names account for summ periods
        # if not bucket_contains_object(sub, obj_name):
        if True:

            print("creating " + stream + " summary file for " + sub + "...")

            # if no label is available for the current date, write a histogram of zeros and continue
            # if not date in label_dates:

            #     print("no label data available for {}; writing zeros".format(date))

            #     write_empty_histogram_for_date(sub, stream, date, experiment_type, bins=10)
            #     continue

            #FIXME: needs to be rewritten to account for UTC offset issue
            # if target data is actually available for the date, create the summary; else, write zeros
            label_data = data_retriever.load_data(cc, sub, label_name, field, label_name, [date])

            if len(label_data) >= 1:

                label_point = label_data[0]

                print("label_point: {}".format(label_point))

                summarization_start = label_point.start_time - dt.timedelta(days=period_days)
                summarization_days = data_retriever.dates_for_stream_between_start_and_end_times(summarization_start, label_point.start_time)
            
                print("summarization days: {}".format(list(summarization_days)))

                summary_grid = []
                quality_list = []
                
                for i in range(0, len(summarization_days)):

                    summ_day = summarization_days[i]

                    biomarker_data = data_retriever.load_data(cc, sub, stream, field, label_name, [summ_day])

                    if len(biomarker_data) == 0:

                        # added this line -- is it unnecessary?
                        write_empty_histogram_for_date(sub, stream, date, experiment_type, bins=10)

                        # no marker data available for the day -- append a 0 to the quality matrix and move on
                        quality_list.append(0)
                        continue

                    marker_keys, marker_groups = dev.group_point_data_by_grid_cell(biomarker_data)
                    marker_grid = dev.project_group_average_onto_grid(stream, marker_keys, marker_groups, dev.x_hour_list_of_empty_y_minute_windows(5))

                    if i == len(summarization_days) - 1:
                        # label day: project target onto day grid up to timestamp
                        label_grid = dev.project_target_onto_grid(label_name, label_point)

                        # filter marker grid by label grid
                        labeled_marker_grid = dev.filter_grid_by_grid(marker_grid, label_grid, "!=", None)

                    else:
                        # prior day completely covered by label: no need to filter 
                        labeled_marker_grid = marker_grid

                    labeled_marker_grid  = dev.collapse_grid(labeled_marker_grid)

                    summary_grid.extend(labeled_marker_grid)
                    quality_list.append(1)

                summary_grid = np.asarray(summary_grid)
                summary_grid = np.reshape(summary_grid, (len(summary_grid), 1))

                q = sum(quality_list) / len(quality_list)
                summary = histogram_from_grid_with_quality_score(summary_grid, q)
                date_keys = [date]

                print("writing histogram: {} with quality score {}".format(list(summary), q))
                write_histograms_to_minio(summary, date_keys, sub, stream, experiment_type, bins)

            else:
                print("warning: found {} values for stream {} on day {}, writing zeros".format(len(label_data), stream, date))
                write_empty_histogram_for_date(sub, stream, date, experiment_type, bins=10)

def write_initial_stream_histogram(sub, stream, stream_id, field, label_name, experiment_type, bins=10):
    """
    Creates an IGTB histogram and writes it to minio.

    Args:
        sub (str): uuid of subject
        stream (str): The stream to summarize
        stream_id (str): uuid of stream to summarize
        field (str): The field within a compound sample to retrieve
        label_name (str): Name of the target stream
        experiment_type (str): Daily vs intake
        bins (int): Number of bins to histogram data into
    """

    obj_name = "{}-{}-{}-{}".format("histogram", bins, stream, "intake")

    if not bucket_contains_object(sub, obj_name):
        biomarker_data = data_retriever.load_data(cc, sub, stream, field, label_name, days="all")

        if len(biomarker_data) == 0:
            write_empty_histogram_for_date(sub, stream, "intake", experiment_type, bins=10)

            # print("no data available for stream {} and user {} on {}: writing zeros!".format(stream, sub, date))

            return
        
        print("marker data example: {}".format(biomarker_data[0]))

        marker_data = [x.sample for x in biomarker_data if x.sample * 0 == 0]

        summary = histogram_from_grid_with_quality_score(marker_data, 1)
        write_histograms_to_minio(summary, ["intake"], sub, stream, experiment_type, bins)

def write_stream_histograms_for_subject(sub, streams, label_name, experiment_type, bins=10):
    """
    Directs execution to the correct function for writing histograms from single (vs compound)
    streams according to experiment type (daily version initial/intake).

    Args:
        sub (str): uuid of subject
        streams (List(str)): The streams to summarize
        label_name (str): Name of the target stream
        experiment_type (str): Daily vs intake
        bins (int): Number of bins to histogram data into
    """
    print("write_stream_histograms_for_subject()")

    available_dates = data_retriever.available_dates_for_user(cc, sub)

    print("available dates for subject: {}".format(list(available_dates)))

    user_streams = cc.get_user_streams(sub)

    print("STARTING SUBJECT {} with dates {}".format(sub, list(available_dates)))

    if not minioClient.is_bucket(sub):
        minioClient.create_bucket(sub)

    for s in streams:

        stream_with_field = s["name"]

        stream, field = mu.stream_and_field(stream_with_field)

        if not stream in user_streams:
            print("stream {} not found in user streams for subject {}!  writing empty histograms!".format(stream, sub))

            for d in available_dates:
                write_empty_histogram_for_date(sub, stream, d, experiment_type, bins)

            continue

        stream_ids = cc.get_stream_id(sub, stream)

        for id in stream_ids:
            
            stream_id = id["identifier"]

            if experiment_type == "daily":
                write_daily_stream_histograms(sub, stream, stream_id, field, available_dates, label_name, experiment_type)

            elif experiment_type == "intake":
                write_initial_stream_histogram(sub, stream, stream_id, field, label_name, experiment_type, bins)

            else:
                print("unrecognized experiment type: {}".format(experiment_type))

def write_daily_compound_histogram(sub, compound_stream, target, label_name, experiment_type, user_dates, period_days=1, bins=10):
    """
    Builds a daily compound histogram and writes the result to minio.

    Args:
        sub (str): uuid of subject
        compound_stream (dict): Dictionary of params describing the compound stream
        label_name (str): Name of the target stream
        experiment_type (str): Daily vs intake
        user_dates (List(str)): String representations of all available data-collection dates for the user
        period_days (int): Number of days to summarize for each label
        bins (int): Number of bins to histogram data into
    """
    if not compound_stream["function"] == "filter":
        print("only compound filter allowed; skipping...")
        return

    compound_params = compound_stream["params"]

    target_stream = compound_params["target-stream"]["name"]
    filter_stream = compound_params["filter-stream"]["name"]

    threshold_value = compound_params["threshold-value"]
    threshold_type = compound_params["threshold-type"]

    # filename to store
    compound_stream_name = "{}-{}-{}-{}-{}".format(target_stream, filter_stream, "filter", threshold_type, threshold_value)
    # obj_name = "{}-{}-{}-{}".format("histogram", bins, compound_stream_name, str(int(date)))

    # get data only for days on which labels are available
    user_streams = cc.get_user_streams(sub)

    label_dates = []
    label_ids = cc.get_stream_id(sub, label_name)

    if (not target_stream in user_streams) and (filter_stream in user_streams) and (label_name in user_streams):
        print("insufficient streams available to write daily compound streams: writing empty histograms")

        for date in user_dates:
            obj_name = "{}-{}-{}-{}".format("histogram", bins, compound_stream_name, str(int(date)))
            write_empty_histogram_for_date(sub, compound_stream_name, date, experiment_type, bins=10)

        return

    for id in label_ids:
        label_uuid = id["identifier"]
        for d in data_retriever.available_dates_for_stream(cc, label_uuid):
            if not d in label_dates:
                label_dates.append(d)
    
    for date in user_dates:

        if not bucket_contains_object(sub, obj_name):
            print("creating compound summary file " + obj_name + " for " + sub + "...")

            # if there's no label, write zeros and move on
            if not date in label_dates:

                print("no label available for stream {} and user {} on {}: writing zeros!".format(compound_stream_name, sub, date))
                write_empty_histogram_for_date(sub, compound_stream_name, date, experiment_type, bins=10)
                continue

            label_data = data_retriever.load_data(cc, sub, label_name, label_name, [date])

            if len(label_data) == 1:

                label_point = label_data[0]
                summarization_start = label_point.start_time - dt.timedelta(days=period_days)
                summarization_days = data_retriever.dates_for_stream_between_start_and_end_times(summarization_start, label_point.start_time)
            
                print("summarization days: {}".format(list(summarization_days)))

                summary_grid = []
                quality_list = []
                
                for i in range(0, len(summarization_days)):

                    summ_day = summarization_days[i]

                    # label data is available: retrieve target and filter streams
                    target_data = data_retriever.load_data(cc, sub, target_stream, target, [summ_day])
                    filter_data = data_retriever.load_data(cc, sub, filter_stream, target, [summ_day])
                    
                    # if there's no target or filter data available, write zeros and move on
                    if (len(target_data) == 0) or (len(filter_data) == 0):
                        quality_list.append(0)
                        write_empty_histogram_for_date(sub, compound_stream_name, date, experiment_type, bins=10)
                        print("insufficient data available for stream {} and user {} on {}: writing zeros!".format(compound_stream_name, sub, date))
                        continue

                    # project target data into grid
                    target_keys, target_groups = dev.group_point_data_by_grid_cell(target_data)
                    target_grid = dev.project_group_average_onto_grid(target_stream, target_keys, target_groups, dev.x_hour_list_of_empty_y_minute_windows(5))

                    # project filter data into grid
                    filter_keys, filter_groups = dev.group_point_data_by_grid_cell(filter_data)
                    filter_grid = dev.project_group_average_onto_grid(filter_stream, filter_keys, filter_groups, dev.x_hour_list_of_empty_y_minute_windows(5))
                    
                    compound_grid = dev.filter_grid_by_grid(target_grid, filter_grid, threshold_type, threshold_value)

                    if i == len(summarization_days) - 1:
                        # label day: project target onto day grid up to timestamp
                        label_grid = dev.project_target_onto_grid(label_name, label_point)

                        # filter marker grid by label grid
                        labeled_compound_grid = dev.filter_grid_by_grid(compound_grid, label_grid, "!=", None)

                    else:
                        # no need to filter 
                        labeled_compound_grid = compound_grid

                    labeled_compound_grid  = dev.collapse_grid(labeled_compound_grid)

                    summary_grid.extend(labeled_compound_grid)
                    quality_list.append(1)

                q = sum(quality_list) / len(quality_list)

                summary_grid = np.asarray(summary_grid)
                summary_grid = np.reshape(summary_grid, (len(summary_grid), 1))

                labeled_compound_hist = histogram_from_grid_with_quality_score(summary_grid, q)
                write_histograms_to_minio(labeled_compound_hist, [date], sub, compound_stream_name, experiment_type, bins)
        
            else:
                print("no label available for {}, writing zeros".format(date))
                write_empty_histogram_for_date(sub, compound_stream_name, date, experiment_type, bins=10)

def write_initial_compound_histogram(sub, compound_stream_params, target, label_name, experiment_type, available_dates, bins=10):
    """
    Builds an IGTB compound histogram and writes the result to minio.

    Args:
        sub (str): uuid of subject
        compound_stream_params (dict): Dictionary of params describing the compound stream
        label_name (str): Name of the target stream
        experiment_type (str): Daily vs intake
        user_dates (List(str)): String representations of all available data-collection dates for the user
        period_days (int): Number of days to summarize for each label
        bins (int): Number of bins to histogram data into
    """
    if not compound_stream_params["function"] == "filter":
        print("only compound filter allowed; skipping...")
        return

    compound_params = compound_stream_params["params"]

    target_stream_with_field = compound_params["target-stream"]["name"]
    filter_stream_with_field = compound_params["filter-stream"]["name"]

    target_stream, target_field = mu.stream_and_field(target_stream_with_field)
    filter_stream, filter_field = mu.stream_and_field(filter_stream_with_field)

    threshold_value = compound_params["threshold-value"]
    threshold_type = compound_params["threshold-type"]

    # filename to store
    compound_stream_name = "{}-{}-{}-{}-{}".format(target_stream, filter_stream, "filter", threshold_type, threshold_value)
    obj_name = "{}-{}-{}-{}".format("histogram", bins, compound_stream_name, "intake")

    if not bucket_contains_object(sub, obj_name):

        print("creating compound summary file " + obj_name + " for " + sub + "...")

        target_dates = []

        user_streams = cc.get_user_streams(sub)

        if not target_stream in user_streams:
            print("no target data available: writing empty arrays")
            write_empty_histogram_for_date(sub, compound_stream_name, "intake", experiment_type, bins=10)

            return

        target_ids = cc.get_stream_id(sub, target_stream)

        for id in target_ids:
            target_uuid = id["identifier"]
            for d in data_retriever.available_dates_for_stream(cc, target_uuid):
                if not d in target_dates:
                    target_dates.append(d)

        summary_grid = []
        quality_list = []

        for i in range(0, len(target_dates)):

            target_day = target_dates[i]

            # label data is available: retrieve target and filter streams
            target_data = data_retriever.load_data(cc, sub, target_stream, target_field, target, [target_day])
            filter_data = data_retriever.load_data(cc, sub, filter_stream, filter_field, target, [target_day])
            
            # if there's no target or filter data available, write zeros and move on
            if (len(target_data) == 0) or (len(filter_data) == 0):
                quality_list.append(0)
                write_empty_histogram_for_date(sub, compound_stream_name, "intake", experiment_type, bins=10)
                print("insufficient data available for stream {} and user {} on {}: writing zeros!".format(compound_stream_name, sub, target_day))
                continue

            # project target data into grid
            target_keys, target_groups = dev.group_point_data_by_grid_cell(target_data)
            target_grid = dev.project_group_average_onto_grid(target_stream, target_keys, target_groups, dev.x_hour_list_of_empty_y_minute_windows(5))

            # project filter data into grid
            filter_keys, filter_groups = dev.group_point_data_by_grid_cell(filter_data)
            filter_grid = dev.project_group_average_onto_grid(filter_stream, filter_keys, filter_groups, dev.x_hour_list_of_empty_y_minute_windows(5))
            
            compound_grid = dev.filter_grid_by_grid(target_grid, filter_grid, threshold_type, threshold_value)

            # for igtb, all days are covered by the target label
            labeled_compound_grid = compound_grid
            labeled_compound_grid  = dev.collapse_grid(labeled_compound_grid)

            summary_grid.extend(labeled_compound_grid)
            quality_list.append(1)

        q = sum(quality_list) / len(quality_list)

        summary_grid = np.asarray(summary_grid)
        summary_grid = np.reshape(summary_grid, (len(summary_grid), 1))

        labeled_compound_hist = histogram_from_grid_with_quality_score(summary_grid, q)
        write_histograms_to_minio(labeled_compound_hist, ["intake"], sub, compound_stream_name, experiment_type, bins)
        
def write_compound_histograms_for_subject(sub, compound_streams, target, label_name, experiment_type, bins=10):
    """
    Directs execution of compound summarization according to experiment type (daily vs initial/intake).

    Args:
        sub (str): uuid of subject
        compound_streams List((dict)): List of dictionaries of params describing the compound stream
        target (dict): Stream to target for filtering or other contextualization
        label_name (str): Name of the Qualtrics label stream
        experiment_type (str): Daily vs intake
        bins (int): Number of bins to histogram data into
    """
    
    available_dates = data_retriever.available_dates_for_user(cc, sub)

    for compound_stream in compound_streams:

        if experiment_type == "daily":
            write_daily_compound_histogram(sub, compound_stream, target, label_name, experiment_type, available_dates, bins=10)

        elif experiment_type == "intake":
            write_initial_compound_histogram(sub, compound_stream, target, label_name, experiment_type, available_dates, bins=10)

def write_summaries_for_subs_streams_target_start_end(json_string):
    """
    Begin execution of a summarization job.  Start point for a Spark job.

    Args:
        json_string (str): JSON string of parameters representing the summarization job
    """
    obj_string = StringIO(json_string)
    obj = json.load(obj_string)
    subjects = obj["subjects"]
    streams = obj["streams"]
    target = obj["target"]
    experiment_type = obj["experiment-type"]
    bins = 10

    label_name = target["name"]

    print("computing summaries for subjects {}".format(list(subjects)))

    for sub in subjects:

        # write single-stream histograms
        write_stream_histograms_for_subject(sub, streams, label_name, experiment_type, bins)

        # write compound histograms
        if "compound-streams" in obj:
            if len(obj["compound-streams"]) > 0:
                compound_streams = obj["compound-streams"]
                write_compound_histograms_for_subject(sub, compound_streams, target, label_name, experiment_type)

    return 1

def write_empty_histogram_for_date(sub, stream_name, date, experiment_type, bins):
    """
    Writes a histogram of all zeroes (including quality score) for dates where no data/label
    is available.

    Args:
        sub (str): uuid of subject
        stream_name (str): Name of marker stream being summarized
        date (str): String representation of the date being summarized
        experiment_type (str): Type of experiment being run (daily vs initial/intake)
        bins (int): Number of bins used for the histogram
    """
    if not minioClient.is_bucket(sub):
            minioClient.create_bucket(sub)

    summary_io = BytesIO()

    y = date + " " + " ".join([str(0)] * (bins + 1)) # +1 for validity score
    summary_io.write(bytes(y, 'utf-8'))
    summary_io.flush()
    summary_io.seek(0)

    if experiment_type == "daily":

        filename = "{}-{}-{}-{}".format("histogram", bins, stream_name, date)

        write_object_to_minio(sub, filename, summary_io)

        summary_io.close()

    elif experiment_type == "intake":
        filename = "{}-{}-{}-{}".format("histogram", bins, stream_name, experiment_type)

        print("writing empty file " + filename + " to bucket " + sub)

        write_object_to_minio(sub, filename, summary_io)
        summary_io.close()

def bucket_contains_object(bucket, object_name):
    
    if not minioClient.is_bucket(bucket):
        print("bucket {} doesn't exist!".format(bucket))
        return False

    file_exists = False

    bucket_files = mC.list_objects(bucket)

    for f in bucket_files:
        f_name = f.object_name.encode('utf-8') # use with CC

        if f_name == bytes(object_name, 'utf-8'):
            file_exists = True
            break
    
    return file_exists

def print_all_buckets_and_objects():
    for b in mC.list_buckets():
        print("bucket {}: ".format(b))

        for o in mC.list_objects(b):
            print("object: {}".format(o.object_name.encode('utf-8')))

def get_np_array_from_minio(bucket, object_name):

    # print(bucket, object_name)

    try:
        data = mC.get_object(bucket, object_name)

        with io.BytesIO() as file_data:
            for d in data.stream(32*1024):
                file_data.write(d)

            file_data.seek(0)

            return np.genfromtxt(file_data, delimiter=" ")

    except ResponseError as err:
        print(err)     

def write_histograms_to_minio(summary, date_keys, subject, stream, experiment_type, bins):

    if experiment_type == "daily":
        for i in range(0, len(date_keys)):

            summary_io = BytesIO()
            y = str(date_keys[i]) + " " + " ".join([str(num) for num in summary[i]])

            # print("y: " + y)

            summary_io.write(bytes(y, 'utf-8'))
            summary_io.flush()
            summary_io.seek(0)

            filename = "{}-{}-{}-{}".format("histogram", bins, stream, date_keys[i])

            print("writing file " + filename + " to bucket " + subject)

            if not minioClient.is_bucket(subject):
                minioClient.create_bucket(subject)

            try:
                mC.put_object(subject, filename, summary_io, len(summary_io.getvalue()))

            except ResponseError as resp_err:
                print(resp_err)

            summary_io.close()

    elif experiment_type == "intake":
        summary_io = BytesIO()
        y = str(0) + " " + " ".join([str(num) for num in summary[0]])

        # print("y: " + y)

        summary_io.write(bytes(y, 'utf-8'))
        summary_io.flush()
        summary_io.seek(0)

        filename = "{}-{}-{}-{}".format("histogram", bins, stream, "intake")

        # print("writing file " + filename + " to bucket " + subject)

        if not minioClient.is_bucket(subject):
            minioClient.create_bucket(subject)

        try:
            mC.put_object(subject, filename, summary_io, len(summary_io.getvalue()))

        except ResponseError as resp_err:
            print(resp_err)

        summary_io.close()

def write_object_to_minio(bucket, object_name, obj):

    if not minioClient.is_bucket(bucket):
        minioClient.create_bucket(bucket)

    try:
        mC.put_object(bucket, object_name, obj, len(obj.getvalue()))

    except ResponseError as resp_err:
        print(resp_err)

def get_available_dates(subjects, streams):
    all_dates = []

    for sub in subjects:

        if not minioClient.is_bucket(sub):
            minioClient.create_bucket(sub)

        if not cc.is_user(sub):
            print("summarizer.get_available_dates: no records found for user {}!".format(sub))
            continue

        else:
            print("summarizer.get_available_dates: found user {}".format(sub))

        user_dates = data_retriever.available_dates_for_user(cc, sub, streams)

        for ud in user_dates:
            if not ud in all_dates:
                all_dates.append(ud)

    return all_dates

def histogram_from_grid_with_quality_score(grid, q, number_of_bins=10):
    """
    Creates a histogram from an array of values using numpy's histogram() function.

    Args:
        grid (numpy array): Array of values from which to create the histogram
        q (int): Quality score for the histogram
        number_of_bins (int): Number of bins used to create the histogram

    Returns:
        h (numpy array): The histogram with quality score prepended
    """

    g = np.asarray(grid)
    g = np.reshape(g, (len(g), 1))
    h = np.histogram(g, bins=number_of_bins)
    h = h[0]

    h = np.append(np.asarray([q]), h, axis=0)

    h = np.reshape(h, (1, len(h)))

    print("histogram_from_grid...(): h: {}, q: {}".format(h, [q]))

    return h
