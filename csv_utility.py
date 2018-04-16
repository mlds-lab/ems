
import numpy as np
import glob
import shutil
import csv


def write_csv_daily(add_name, participant, date, time_zone, score_name, score_estimate):
    """
    Write the score estimates for variable score_name (DGTB) to a csv file.

    Args:
        add_name (str): prefix to be added to the beginning of output file names.
        participant (numpy array): array of participantIDs
        date (numpy array): array of dates
        time_zone (numpy array): array of time zones
        score_name (str): score name
        score_estimate (numpy array): array of estimates for data cases

    Returns:
        output file name (str)

    """

    indices = np.isfinite(score_estimate)

    p, q = np.min(score_estimate[indices]), np.max(score_estimate[indices])
    n_rows = len(participant)

    d = {'stress.d': [1, 5],
         'anxiety.d': [1, 5],
         'pos.affect.d': [5, 25],
         'neg.affect.d': [5, 25],
         'irb.d': [7, 49],
         'itp.d': [1, 5],
         'ocb.d': [0, 8],
         'cwb.d': [0, 8],
         'sleep.d': [0, 24],
         'alc.quantity.d': [0, q],
         'tob.quantity.d': [0, q],
         'total.pa.d': [0, q],
         'neuroticism.d': [1, 5],
         'conscientiousness.d': [1, 5],
         'extraversion.d': [1, 5],
         'agreeableness.d': [1, 5],
         'openness.d': [1, 5]}

    if score_name in d.keys():
        score_range = d[score_name]
    else:
        raise ValueError('Score name invalid!')

    # range map:
    score_estimate[indices] = np.interp(score_estimate[indices], [p, q], score_range)

    score_estimate1 = np.full(score_estimate.shape, 'NA', dtype='U20')
    score_estimate1[indices] = score_estimate[indices].astype('U20')

    # fixing the format of dates
    date = np.array(date, dtype=object)
    for i in range(date.shape[0]):
        x = date[i]
        m, d, y = x[4:6], x[6:], x[:4]
        d = str(int(d))
        date[i] = m + '/' + d + '/' + y

    # defining empty startTime and endTime arrays
    start_time = np.repeat('', n_rows)
    end_time = np.repeat('', n_rows)

    a = np.rec.fromarrays((participant, date, start_time, end_time, time_zone,
                           np.repeat(score_name, n_rows), score_estimate1))

    file_name = add_name + score_name + ".csv"
    np.savetxt(file_name, a, delimiter=",", fmt='%20s, %8s, %s, %s, %3s, %10s, %20s', comments='',
               header='ParticipantID, Date, StartTime, EndTime, Timezone, ScoreName, ScoreEstimate')

    return file_name


def write_csv_initial(add_name, participant, datetime, time_zone, score_name, score_estimate):
    """
    Write the score estimates for variable score_name (IGTB) to a csv file.

    Args:
        add_name (str): prefix to be added to the beginning of output file names.
        participant (numpy array): array of participantIDs
        datetime (numpy array): array of date-times for the data cases
        time_zone (numpy array): array of time zones for the data cases
        score_name (str): score name
        score_estimate (numpy array): array of estimates for data cases

    Returns:
        output file name (str)

    """

    indices = np.isfinite(score_estimate)

    p, q = np.min(score_estimate[indices]), np.max(score_estimate[indices])

    d = {'irb': [7, 49],
         'itp': [1, 5],
         'ocb': [20, 100],
         'inter.deviance': [7, 49],
         'org.deviance': [12, 84],
         'shipley.abs': [0, 25],
         'shipley.vocab': [0, 40],
         'neuroticism': [1, 5],
         'conscientiousness': [1, 5],
         'extraversion': [1, 5],
         'agreeableness': [1, 5],
         'openness': [1, 5],
         'pos.affect': [10, 50],
         'neg.affect': [10, 50],
         'stai.trait': [20, 80],
         'audit': [0, 40],
         'gats.quantity': [0, q],
         'ipaq': [0, q],
         'psqi': [0, 21],
         'gats.status': [0, 5]}

    if score_name in d.keys():
        score_range = d[score_name]
    else:
        raise ValueError('Score name invalid!')

    indices = np.isfinite(score_estimate)

    # range map:
    score_estimate[indices] = np.interp(score_estimate[indices], [p, q], score_range)

    score_estimate1 = np.full(score_estimate.shape, 'NA', dtype='U20')
    score_estimate1[indices] = score_estimate[indices].astype('U20')

    a = np.rec.fromarrays((participant, datetime, time_zone, score_estimate1))

    file_name = add_name + score_name + ".csv"
    np.savetxt(file_name, a, delimiter=",", fmt='%20s, %10s, %s, %20s', comments='',
               header=f'ParticipantID, igtb.datatime, igtb.timezone, {score_name}')

    return file_name


def merge_csv_initial(output_filename, path):
    """"Merge csv files corresponding to different initial variables into a single file.
        Inputs:
        output_filename: the name (and path) of the merged file;
        example: output_filename = 'df_out.csv'
        path: path of the csv files to be merged (import csv files from this folder);
        example: path = r'data/US/market/merged_data'
    """

    prefix = ['ParticipantID',
              'igtb.datatime',
              'igtb.timezone']

    names = ['irb',
             'itp',
             'ocb',
             'inter.deviance',
             'org.deviance',
             'shipley.abs',
             'shipley.vocab',
             'neuroticism',
             'conscientiousness',
             'extraversion',
             'agreeableness',
             'openness',
             'pos.affect',
             'neg.affect',
             'stai.trait',
             'audit',
             'gats.quantity',
             'ipaq',
             'psqi']

    b = np.loadtxt(path + names[0] + '.csv', delimiter=",", skiprows=1, usecols=(0, 1, 2))
    a = np.array(b, dtype=object)

    for n in names:

        b = np.loadtxt(path + n + '.csv', delimiter=",", skiprows=1, usecols=3)
        a = np.rec.fromarrays((a, b))

    column_format = '%20s %10s %10s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f'
    names_string = ','.join(prefix + names)

    np.savetxt(output_filename, a, delimiter=",", fmt=column_format, comments='', header=names_string)

    return output_filename


def merge_csv_daily_cheap(output_filename, path):
    """"Merge csv files corresponding to different daily variables into a single file; without copying to memory.
    Inputs:
    output_filename: the name (and path) of the merged file; example: output_filename = 'df_out.csv'
    path: path of the csv files to be merged (import csv files from this folder);
    example: path = r'data/US/market/merged_data'
    """

    # import csv files from folder
    allFiles = glob.glob(path + "/*.csv")

    with open(output_filename, 'wb') as outfile:
        for i, fname in enumerate(allFiles):
            with open(fname, 'rb') as infile:
                if i != 0:
                    infile.readline()  # Throw away header on all but first file
                # Block copy rest of file from input to output without parsing
                shutil.copyfileobj(infile, outfile)
                print(fname + " has been imported.")

    return output_filename


def merge_csv_daily(output_filename, path):
    """"Merge csv files corresponding to different daily variables into a single file; with copying to memory.
    Inputs:
    output_filename: the name (and path) of the merged file;
    example: output_filename = 'df_out.csv'
    path: path of the csv files to be merged (import csv files from this folder);
    example: path = r'data/US/market/merged_data'
    """

    # import csv files from folder
    allFiles = glob.glob(path + "/*.csv")

    write_headers = True
    with open(output_filename, 'wb') as fout:
        writer = csv.writer(fout)
        for filename in allFiles:
            with open(filename) as fin:
                reader = csv.reader(fin)
                headers = reader.next()
                if write_headers:
                    write_headers = False  # Only write headers once.
                    writer.writerow(headers)
                writer.writerows(reader)  # Write all remaining rows.
                print(filename + " has been imported.")

    return output_filename
