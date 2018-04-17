import sys
#from features import extract_features
import datetime as dt
import time
import json
import numpy as np
import itertools as it
from sklearn.preprocessing import Imputer
from minio import Minio
from minio.error import ResponseError
import socket

if socket.gethostname() == "cerebralcortex":
    minioClient = Minio('127.0.0.1:9000',
        access_key='ZngmrLWgbSfZUvgocyeH',
        secret_key='IwUnI5w0f5Hf1v2qVwcr',
        secure=False)
else:
    minioClient = None

positions_file = open('field_positions.json', 'w+')

if positions_file.tell() == 0:
    field_positions = {}

else:
    field_positions = json.load(field_positions)

def get_parameters(exp):
    """
    Convenience function for unpacking the EDD

    :param dict exp: Dictionary of objects that describe the experiment

    :return: Unpacked set of objects describing the parameters of the experiment
    :rtype: Various
    """
    experiment_type = exp["experiment-type"]
    streams = exp["marker-streams"]
    target = exp["target"]
    models = exp["models"]
    loss_function = exp["loss-function"]
    data_range = exp["data-range"]
    start_time = data_range["start-time"]
    end_time = data_range["end-time"]
    wb_list = exp["white-black-list"]

    if not start_time == "all":
        print("start and end times not supported yet: defaulting to 'all'")
        
        start_time = "all"
        end_time = ""

    if "prediction" in exp:
        prediction = exp["prediction"]
        predict_subjects = prediction["subjects"]
        predict_data_range = prediction["data-range"]

        predict_start_time = predict_data_range["start-time"]
        predict_end_time = predict_data_range["end-time"]

        if not predict_start_time == "all":
            print("start and end times not supported yet: defaulting to 'all'")

            predict_start_time = "all"
            predict_end_time = ""
            
    else:
        predict_subjects = []
        predict_data_range = ""

    if "compound-streams" in exp:
        compound_streams = exp["compound-streams"]
    else:
        compound_streams = {}

    return experiment_type, streams, compound_streams, target, models, loss_function, start_time, end_time, wb_list, predict_subjects, predict_start_time, predict_end_time

def stream_and_field(stream_with_field):

    stream_and_field = stream_with_field.split("&")

    if len(stream_and_field) == 1:
        return stream_with_field, ""

    stream_name = stream_and_field[0]
    field_name = stream_and_field[1]

    return stream_name, field_name

def print_progress(message):
    sys.stdout.write(message)
    sys.stdout.flush()

# def clean(data):
#     imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
#     imp.fit(data)
#     return imp.transform(data)

# def sort(data):
#     return np.array(sorted(data, key=lambda x: x[1]))

# def group_by_activity(data):
#     keys = []
#     groups = []
#     activity_type_index = 1

#     for k, g in it.groupby(data, lambda x: x[activity_type_index]):
#         keys.append(k)
#         groups.append(list(g))

#     return keys, groups

# def group_by_date(data):
	
#     #FIXME: temporary until switching to CC!!
#     print("mu.group_by_date() data: {}".format(list(data)))
#     # return group_by_minute(data)

#     keys = []
#     groups = []
#     timestamp_index = 0

#     for k, g in it.groupby(data, lambda x: x.start_time.date):

#         keys.append(k)
#         groups.append(list(g))

#     return keys, groups

# def group_by_minute(data):
#     keys = []
#     groups = []
#     timestamp_index = 0

#     stored_time = 1501182103867 / (1000)

#     # FIXME: this is ugly but temporary (PAMAP2-specific)
#     for k, g in it.groupby(data, lambda x: time.mktime(time.strptime(dt.datetime.fromtimestamp(stored_time + (x[timestamp_index])).strftime("%Y-%m-%d-%H-%M"), "%Y-%m-%d-%H-%M")) * 1000):
#         keys.append(k)
#         groups.append(list(g))

#     return keys, groups

# def group_by_x_minutes(data, minutes):
#     keys = []
#     groups = []
#     timestamp_index = 0

#     print("mu.group_by_x_minutes() data: {}".format(list(data)))

#     # FIXME: this is ugly
#     for k, g in it.groupby(data, lambda x: time.mktime(time.strptime(dt.datetime.fromtimestamp(x[timestamp_index]).strftime("%Y-%m-%d-%H-%M")), "%Y-%m-%d-%H-%M") * 1000 + 1000 * minutes):
#         keys.append(k)
#         groups.append(list(g))

#     return keys, groups

# def window(data):
#     X = np.zeros((0, len(data[0]) - 2))
#     y = np.zeros(0,)
#     for i, labeled_window in slidingWindow(data, 1000, 200):
#         naked_data = labeled_window[:, 2:]
#         x = extract_features(naked_data)
#         x = np.reshape(x, (1, -1))
#         X = np.append(X, x, axis=0)
#         y = np.append(y, labeled_window[0, 1])

#     return X, y

def date_array_of_x_minute_windows(start_date, window_size_minutes):
    """
    Returns a list of datetime objects representing a 24-hour period in windows of temporal length
    window_size_minutes

    :param datetime start_date: datetime representing the start of the 24-hour period
    :param int window_size_minutes: temporal length of the windows that the 24-hour period are
        being broken up into

    :return: List of datetime objects
    :rtype: List(datetime)
    """
    day_array = []
    minutes_per_day = 24 * 60
    boxes_per_day = int(minutes_per_day / window_size_minutes)

    current_datetime = start_date

    for i in range(0, boxes_per_day):
        current_datetime = current_datetime + dt.timedelta(minutes=5)
        # current_datetime.minute = current_datetime.minute + window_size_minutes
        day_array.append(current_datetime)

    return day_array

# def slidingWindow(sequence,winSize,step=1):
#     """
#     Returns a generator that will iterate through
#     the defined chunks of input sequence.  Input sequence
#     must be iterable.
#     Thanks to Sean Noran and https://scipher.wordpress.com/2010/12/02/simple-sliding-window-iterator-in-python/
#     """
 
#     # Verify the inputs
#     try: 
#         it = iter(sequence)
#     except TypeError:
#         raise Exception("**ERROR** sequence must be iterable.")
#     if not ((type(winSize) == type(0)) and (type(step) == type(0))):
#         raise Exception("**ERROR** type(winSize) and type(step) must be int.")
#     if step > winSize:
#         raise Exception("**ERROR** step must not be larger than winSize.")
#     if winSize > len(sequence):
#         raise Exception("**ERROR** winSize must not be larger than sequence length.")
 
#     # Pre-compute number of chunks to emit
#     numOfChunks = ((len(sequence)-winSize)/step)+1
 
#     # Do the work
#     for i in range(0,numOfChunks*step,step):
#         yield i, sequence[i:i+winSize]

def print_all_minio_objects():
    # minioClient = Minio('127.0.0.1:9000',
    #               access_key='ZngmrLWgbSfZUvgocyeH',
    #               secret_key='IwUnI5w0f5Hf1v2qVwcr',
    #               secure=False)
    if minioClient:
        for bucket in minioClient.list_buckets():
            print(str(bucket))

            for o in minioClient.list_objects(bucket.name):
                print(str(o))

    else:
        print("minioClient not available in production environment!")

def clear_minio_data():
    # minioClient = Minio('127.0.0.1:9000',
    #               access_key='ZngmrLWgbSfZUvgocyeH',
    #               secret_key='IwUnI5w0f5Hf1v2qVwcr',
    #               secure=False)

    if minioClient:
        for bucket in minioClient.list_buckets():
            for o in minioClient.list_objects(bucket.name):
                print("removing {} in bucket {}".format(o, bucket.name))
                minioClient.remove_object(bucket.name, o.object_name)

            minioClient.remove_bucket(bucket.name)

        print_all_minio_objects()
        
    else:
        print("minioClient not available in production environment!")
