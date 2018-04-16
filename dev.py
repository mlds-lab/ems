import datetime as dt
import pytz
import time
import numpy as np
import itertools as it
import math as math
import numbers
import json

STREAM_MAPPINGS = {
        "org.md2k.data_qualtrics.feature.v10.stress.d" : 'stress.d',
        "org.md2k.data_qualtrics.feature.v10.agreeableness.d": 'agreeableness.d',
        "org.md2k.data_qualtrics.feature.v10.alc.quantity.d": 'alc.quantity.d',
        "org.md2k.data_qualtrics.feature.v10.anxiety.d": 'anxiety.d',
        "org.md2k.data_qualtrics.feature.v10.conscientiousness.d": 'conscientiousness.d',
        "org.md2k.data_qualtrics.feature.v10.cwb.d": 'cwb.d',
        "org.md2k.data_qualtrics.feature.v10.extraversion.d": 'extraversion.d',
        "org.md2k.data_qualtrics.feature.v10.irb.d": 'irb.d',
        "org.md2k.data_qualtrics.feature.v10.itp.d": 'itp.d',
        "org.md2k.data_qualtrics.feature.v10.neg.affect.d": 'neg.affect.d',
        "org.md2k.data_qualtrics.feature.v10.neuroticism.d": 'neuroticism.d',
        "org.md2k.data_qualtrics.feature.v10.ocb.d": 'ocb.d',
        "org.md2k.data_qualtrics.feature.v10.openness.d": 'openness.d',
        "org.md2k.data_qualtrics.feature.v10.pos.affect.d": 'pos.affect.d',
        "org.md2k.data_qualtrics.feature.v10.sleep.d": 'sleep.d',
        "org.md2k.data_qualtrics.feature.v10.tob.quantity.d": 'tob.quantity.d',
        "org.md2k.data_qualtrics.feature.v10.total.pa.d": 'total.pa.d',

        "org.md2k.data_qualtrics.feature.v11.igtb.agreeableness": 'agreeableness',
        "org.md2k.data_qualtrics.feature.v11.igtb.audit": 'audit',
        "org.md2k.data_qualtrics.feature.v11.igtb.conscientiousness": 'conscientiousness',
        "org.md2k.data_qualtrics.feature.v11.igtb.extraversion": 'extraversion',
        "org.md2k.data_qualtrics.feature.v11.igtb.gats.quantity": 'gats.quantity',
        "org.md2k.data_qualtrics.feature.v11.igtb.gats.status": 'gats.status',
        "org.md2k.data_qualtrics.feature.v11.igtb.inter.deviance": 'inter.deviance',
        "org.md2k.data_qualtrics.feature.v11.igtb.ipaq": 'ipaq',
        "org.md2k.data_qualtrics.feature.v11.igtb.irb": 'irb',
        "org.md2k.data_qualtrics.feature.v11.igtb.itp": 'itp',
        "org.md2k.data_qualtrics.feature.v11.igtb.neg_effect": 'neg.affect',
        "org.md2k.data_qualtrics.feature.v11.igtb.neuorticism": 'neuroticism',
        "org.md2k.data_qualtrics.feature.v11.igtb.ocb": 'ocb',
        "org.md2k.data_qualtrics.feature.v11.igtb.openness": 'openness',
        "org.md2k.data_qualtrics.feature.v11.igtb.org_deviance": 'org.deviance',
        "org.md2k.data_qualtrics.feature.v11.igtb.pos.affect": 'pos.affect',
        "org.md2k.data_qualtrics.feature.v11.igtb.psqi": 'psqi',
        "org.md2k.data_qualtrics.feature.v11.igtb.shipley.abs": 'shipley.abs',
        "org.md2k.data_qualtrics.feature.v11.igtb.shipley.vocab": 'shipley.vocab',
        "org.md2k.data_qualtrics.feature.v11.igtb.stai.trait": 'stai.trait'
    }

def array_of_timestamps(start_time, end_time, window_size_ms):
    """
    Creates and returns an array of timestamps.

    Args:
        start_time (datetime): Starting time of first timestamp
        end_time (datetime): Time of last timestamp
        window_size_ms (int): Number of milliseconds between timestamps

    Returns:
        a (numpy array): The array of timestamps
    """
    d = end_time - start_time
    m = d.minutes
    windows = m / (window_size_ms / 1000)

    a = [-1 for i in range(0, windows)]

    for i in range(0, windows):
        a[i] = start_time + i * window_size_ms # FIXME: does this work?

    return np.asarray(a)

def x_hour_list_of_empty_y_minute_windows(window_size_minutes=5, grid_size_hours=24):
    """
    Creates a list of empty indices.

    Args:
        window_size_minutes (int): The number of minutes between windows
        grid_size_hours (int): The number of hours the grid will represent

    Returns:
        grid (List): An empty list of length grid_size_hours * 60 / window_size_minutes
    """
    total_minutes = grid_size_hours * 60
    windows_per_day = int(total_minutes / window_size_minutes)

    grid = [None] * windows_per_day

    return grid

def date_list_of_x_minute_windows(start_date, window_size_minutes):
    """
    Creates a list of empty indices.

    Args:
        window_size_minutes (int): The number of minutes between windows
        grid_size_hours (int): The number of hours the grid will represent

    Returns:
        grid (List): An empty list of length grid_size_hours * 60 / window_size_minutes
    """
    day_array = []
    minutes_per_day = 24 * 60
    windows_per_day = int(minutes_per_day / window_size_minutes)

    current_datetime = start_date

    for i in range(0, windows_per_day):
        current_datetime = current_datetime + dt.timedelta(minutes=5)
        day_array.append(current_datetime)

    return day_array

def grouping_function(d_point):
    """
    Support function for the call to groupby in group_point_data_by_grid_cell().

    Args:
        d_point (DataPoint): The CerebralCortex DataPoint object to be grouped into a window by timestamp.

    Returns:
        interval_window (int): The window within a temporal grid to which the DataPoint belongs
    """

    #TODO: parameterize window sizes
    window_size_minutes = 5
    
    datapoint = d_point
    datapoint.start_time = datapoint.start_time.replace(tzinfo=None)

    start_time = date_start_from_data_point(datapoint)

    interval = (datapoint.start_time - start_time) / window_size_minutes
    interval_minutes = interval.seconds / 60
    interval_window = int(interval_minutes / window_size_minutes)

    return interval_window


def group_point_data_by_grid_cell(data, start_time=None, window_size_minutes=5):
    """
    Groups a set of data points into windows within a day according their timestamps.

    Args:
        data (List(DataPoint)): The set of DataPoint objects to group by timestamp
        start_time (datetime): The start time of the set
        window_size_minutes (int): The interval between windows within the grid's duration

    Returns:
        keys (List(int)): The integer values of the windows to which data points belong
        groups (List(DataPoint)): The groups of DataPoint objects that belong in the windows in keys
    """

    # print("dev.group_point_data_by_grid_cell() data: {}".format(list(data)))

    keys = []
    groups = []

    if not start_time:
        start_time = date_start_from_data_point(data[0])

    data = sorted(data, key=grouping_function)
    for k, g in it.groupby(data, key=grouping_function):
        keys.append(k)
        groups.append(list(g))

    return keys, groups

def date_start_from_data_point(datapoint):
    """
    Support function that finds the date from a data point.

    Args:
        datapoint (DataPoint): The data point whose date is to be extracted

    Returns:
        start_time (datetime): A timestamp representing the start of the date (12 am) of the data point
    """

    dp_start = datapoint.start_time
    start_time = dt.datetime(dp_start.year, dp_start.month, dp_start.day)
    start_time.replace(tzinfo=dp_start.tzinfo)

    return start_time

def project_group_average_onto_grid(stream_name, keys, groups, grid):
    """
    Takes grouped DataPoint objects, averages their sample values and adds them to a day grid.

    Args:
        stream_name (str): Name of the stream being processed
        keys (List(int)): A list of integer keys representing windows in a day grid
        groups (List(DataPoint)): The data points to be averaged and projected into grid windows
        grid (List): The grid into which the averaged values will be projected

    Returns:
        projected_grid (List(float)): The grid into which the averaged values have been projected
    """
    projected_grid = grid

    for i in range(0, len(keys)):
        key = keys[i]
        group = groups[i]

        # print("i: {}, key: {}, group: {}".format(i, key, group))

        if key < len(grid):
            projected_grid[key] = np.mean([g.sample for g in group])

    return projected_grid

def project_group_max_onto_grid(stream_name, keys, groups, grid):
    """
    Takes grouped DataPoint objects, finds the max of their sample values and adds it to a day grid.

    Args:
        stream_name (str): Name of the stream being processed
        keys (List(int)): A list of integer keys representing windows in a day grid
        groups (List(DataPoint)): The data points to be maxed and projected into grid windows
        grid (List): The grid into which the max values will be projected

    Returns:
        projected_grid (List(float)): The grid into which the max values have been projected
    """
    projected_grid = grid

    # print("len(keys): {}, len(grid): {}".format(len(keys), len(grid)))

    for i in range(0, len(keys)):
        key = keys[i]
        group = groups[i]

        clean_list = [g.sample for g in group]
        if key < len(grid):
                projected_grid[key] = np.max(clean_list)

    return projected_grid

def project_target_onto_grid(target_name, label_point):
    """
    Projects target label DataPoints into a grid for filtering.

    Args:
        target_name (str): The name of the target to be predicted
        label_point (DataPoint): The target value itself to be projected onto the grid

    Returns:
        grid (List): A list into which the target value has been projected
    """
    keys, groups = group_point_data_by_grid_cell([label_point])
    grid = x_hour_list_of_empty_y_minute_windows()
    
    # project target label onto grid for the period *up to* the time of the label
    for i in range(0, keys[0]):
        grid[i] = groups[0]

    return grid

def flood_grid_with_target_values(label_point):
    """
    Fills a grid with target label DataPoints.

    Args:
        label_point (DataPoint): The label to be projected into the grid

    Returns:
        grid (List(DataPoint)): The list into which the label point has been completely projected
    """
    grid = x_hour_list_of_empty_y_minute_windows()
    
    for i in range(0, len(grid)):
        grid[i] = label_point

    return grid

def collapse_grid(grid):
    """
    Removes all non-value elements from a grid.

    Args:
        grid (List(DataPoint)): The grid with a combination of value and None/nan cells

    Returns:
        collapsed_grid (List(DataPoint)): The grid containing only values
    """

    collapsed_grid = [g for g in grid if not g == None]
    return collapsed_grid

def get_label_field_name(stream):
    """
    Utility function to help identify the field within a compound sample.

    Args:
        stream (str): Name of the stream being processed

    Returns:
        STREAM_MAPPINGS[stream] (str): String name of the field within the compound sample
    """
    if stream in STREAM_MAPPINGS:
        # print("field name mapping for {}: {}".format(stream, str(STREAM_MAPPINGS[stream])))
        return STREAM_MAPPINGS[stream]

    else:
        print("no field name mapping for " + stream)

        return None

def filter_grid_by_grid(target_grid, filter_grid, threshold_type, threshold):
    """
    Compares two grids, a target grid (to be filtered) and a filter grid, and keeps values in the target
    grid whenever the corresponding values in the filter grid meet a certain criterion.

    Args:
        target_grid (List(DataPoint)): The values to keep or remove
        filter_grid (List(DataPoint)): The values to compare to a filter criterion (type and threshold)
        threshold_type (str): The type of filtering to be done: <, <=, ==, >=, > or !=
        threshold (val): The value to compare to values in the filter grid according to the threshold type
    Returns:
        ret_grid (List(DataPoint)): The filtered grid
    """

    ret_grid = [None] * len(target_grid)

    for i in range(0, len(target_grid)):

        filter_point = filter_grid[i]
        target_point = target_grid[i]

        # print("filter point: {}, target point: {}, threshold: {}".format(filter_point, target_point, threshold))

        no_target = target_point == None

        # print("no_target: {}".format(no_target))

        threshold_not_None = threshold != None

        # print("threshold_not_None: {}".format(threshold_not_None))

        filter_point_is_None = filter_point is None

        # print("filter_point_is_None: {}".format(filter_point_is_None))
        
        no_filter_not_filtering_on_None = filter_point_is_None and threshold_not_None

        # print("no_filter...: {}".format(no_filter_not_filtering_on_None))

        if no_target or no_filter_not_filtering_on_None:
        # if target_point == None or (filter_point == None and not threshold == None):

            # print("continuing...")

            continue

        elif threshold_type == "<":
            if filter_point < threshold:
                ret_grid[i] = target_grid[i]
                # print("{} {} {}".format(str(filter_point), threshold_type, str(threshold)))

        elif threshold_type == "<=":
            if filter_point <= threshold:
                ret_grid[i] = target_grid[i]
                # print("{} {} {}".format(str(filter_point), threshold_type, str(threshold)))

        elif threshold_type == "==":

            if (threshold is None) and (filter_point is None):
                ret_grid[i] = target_grid[i]

            elif filter_point == threshold:
                ret_grid[i] = target_grid[i]
                # print("{} {} {}".format(str(filter_point), threshold_type, str(threshold)))

        elif threshold_type == ">=":
            if filter_point >= threshold:
                ret_grid[i] = target_grid[i]
                # print("{} {} {}".format(str(filter_point), threshold_type, str(threshold)))

        elif threshold_type == ">":
            if filter_point > threshold:
                ret_grid[i] = target_grid[i]
                # print("{} {} {}".format(str(filter_point), threshold_type, str(threshold)))

        elif threshold_type == "!=":

            if (threshold is None) and (filter_point is not None):
                ret_grid[i] = target_grid[i]

            elif filter_point != threshold:
                ret_grid[i] = target_grid[i]
                # print("{} {} {}".format(str(filter_grid[i]), threshold_type, str(threshold)))

    # print("returning...")

    return ret_grid