import numpy as np
import time
from datetime import datetime
from datetime import date, timedelta
import mosaic_utils as mu

def load_data(cc, subject, marker, field, target=None, days=None, check=False):
    """
    Pass-through function.  Accounts for the possibility of various data sources.

    :param CerebralCortex cc: CerebralCortex instance
    :param str subject: uuid of subject whose data is being retrieved
    :param str marker: Name of marker stream to retrieve
    :param str field: Name of field to return for compound DataPoint.sample types
    :param str target: Name of prediction target, can be used to ignore days in which label
        data isn't available (currently unused)
    :param List(str) days: Explicit list of days to retrieve data for

    :return: List of DataPoint objects
    :rtype: List(DataPoint)
    """
    return load_cc_data(cc, subject, marker, field, target, days)

def load_cc_data(cc, subject, marker, field, target=None, all_days=None, check=False):
    """
    Primary means of retrieving data from CerebralCortex.  Uses CerebralCortex functions
    to retrieve data for a particular subject-stream combination, combining data from
    one or more days into a single list.

    :param CerebralCortex cc: CerebralCortex instance
    :param str subject: uuid of subject whose data is being retrieved
    :param str marker: Name of marker stream to retrieve
    :param str field: Name of field to return for compound DataPoint.sample types
    :param str target: Name of prediction target, can be used to ignore days in which label
        data isn't available (currently unused)
    :param List(str) all_days: Explicit list of days to retrieve data for

    :return: List of DataPoint objects
    :rtype: List(DataPoint)
    """
    full_stream = []

    if all_days == None or all_days == "all":
            print("dr.load_cc_data: setting days to 'all'")
            # all_days = available_dates_for_stream(cc, marker_id)
            all_days = available_dates_for_user_and_stream_name(cc, subject, marker)

    if check:
        streams = cc.get_user_streams(subject)
        if len(streams) > 0 and marker not in streams:
            print("data retriever: stream {} not available for user {}".format(marker, subject))
            return full_stream

    print("found marker {} in streams for user {}".format(marker, subject))

    #FIXME: this can also be handled higher up -- get stream uuids, pass to data retriever
    marker_ids = cc.get_stream_id(subject, marker)

    for id in marker_ids:
        marker_id = id["identifier"]

        # print("data retriever: load_cc_data: {}: {}".format(marker, marker_id))

        if marker_id == None:
            print("marker ID not found")
            continue

        # if all_days == None or all_days == "all":
        #     print("dr.load_cc_data: setting days to 'all'")
        #     all_days = available_dates_for_stream(cc, marker_id)

        for d in all_days:

            # print("getting stream {} for day {}".format(marker_id, d))

            marker_stream = cc.get_stream(marker_id, subject, d)
            full_stream.extend(marker_stream.data)

            # print("found {} values for stream {}".format(len(full_stream), marker))

            return full_stream


def available_dates_for_user(cc, subject_id, streams=None):
    """
    Discovers all dates for which a user might potentially have collected data.  Loops through
    all available uuids for all available streams for the specified user; durations for each
    uuid are gotten from CerebralCortex.get_stream_duration().  These durations are converted to
    lists of explicit date strings using the dates_for_stream_between_start_and_end_times() utility
    function.

    :param CerebralCortex cc: CerebralCortex instance for accessing user streams
    :param uuid subject_id: uuid of subject whose stream durations will be queried for
    :param List(uuid) streams: explicit list of stream uuids to query for durations

    :return: List of all dates in which a might have collected data
    :rtype: List(str)
    """

    all_dates = []

    if not streams:
        streams = cc.get_user_streams(subject_id)

    for s in streams:

        if not (('data_analysis' in s) or ('data_qualtrics' in s)):
            # print("dismissing stream {} from date discovery".format(s))
            continue

        stream_ids = cc.get_stream_id(subject_id, s)

        # print("retriever.available_dates_for_user: user: {}, stream: {}, stream_ids: {}".format(subject_id, s, list(stream_ids)))

        for id in stream_ids:

            stream_id = id["identifier"]

            duration = cc.get_stream_duration(stream_id)

            # print("-" * 30)

            # print("stream {} with uuid {} for user {} duration: {}".format(s, stream_id, subject_id, duration))

            stream_dates = dates_for_stream_between_start_and_end_times(duration["start_time"], duration["end_time"])

            for sd in stream_dates:
                if not sd in all_dates:
                    all_dates.append(sd)

    return all_dates

def available_dates_for_user_and_stream_name(cc, user_id, stream_name, check=False):
    dates = []

    if check:
        user_streams = cc.get_user_streams(user_id)

        if stream_name not in user_streams:
            print("data retriver: stream {} not available for user {}".format(stream_name, user_id))
            return dates

    stream_ids = cc.get_stream_id(user_id, stream_name)

    for id in stream_ids:
        stream_uuid = id["identifier"]

        for d in available_dates_for_stream(cc, stream_uuid):
            if d not in dates:
                dates.append(d)

    return dates

def available_dates_for_stream(cc, stream_id):
    """
    Discovers all available dates within a stream's duration.

    :param CerebralCortex cc: CerebralCortex instance
    :param str stream_id: uuid of stream to retrieve dates for

    :return: Explicit list of string representations of all dates within the given stream's duration
    :rtype: List(str)
    """
    all_days = []

    stream_duration = cc.get_stream_duration(stream_id)

    if stream_duration is None:
        print("no duration data available for stream ID " + str(stream_id))

    else:
        stream_start_time = stream_duration["start_time"]
        stream_end_time = stream_duration["end_time"]

        stream_start = datetime(stream_start_time.year, stream_start_time.month, stream_start_time.day)
        stream_end = datetime(stream_end_time.year, stream_end_time.month, stream_end_time.day)

        stream_interval = stream_end - stream_start

        number_of_days = stream_interval.days + 1 # add 1 to capture first and last days

        for i in range(0, number_of_days):
            all_days.append((stream_start + timedelta(days=i)).strftime("%Y%m%d"))

    return all_days

def dates_for_stream_between_start_and_end_times(stream_start, stream_end):
    """
    Generates list of explicit dates within a stream's duration

    :param datetime stream_start: Start of a stream's duration
    :paramt datetime stream_end: End of a stream's duration

    :return: Explicit list of string representations of dates between two datetime objects
    :rtype: List(str)
    """

    dates = []

    stream_interval = stream_end - stream_start

    number_of_days = stream_interval.days + 1 # add 1 to capture first and last days

    for i in range(0, number_of_days):
        dates.append((stream_start + timedelta(days=i)).strftime("%Y%m%d"))

    return dates