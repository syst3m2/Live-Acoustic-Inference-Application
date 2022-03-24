sample_rate = 8000
n_channels = 4

from collections import namedtuple
from math import floor, ceil

from sqlite_resources import create_connection, execute_sql_command, execute_multiple_sql_command
from data_types import IntegerTimestamp

def get_db_entries(db_file, table, start_t, stop_t):
    # sql = "SELECT * from {} where end_time >=? and start_time <=?;".format(table)
    sql = "SELECT * from {} where end_time_sec >=? and start_time_sec <=?;".format(table) #todo: need to find a safer workaround than string formatting here, but sqlite doesn't allow ? for tables
    conn = create_connection(db_file)
    with conn:
        cur = execute_sql_command(conn, sql, (start_t, stop_t)) #todo: do I need a +1 here?
    return cur

WavReaderInfo = namedtuple('WavReaderInfo', ['filename', 'file_start_time', 'file_end_time', 'start_position', 'end_position'])

def get_named_audio_file_range_info0(entry, start_t=0, stop_t=0):
    entry = list(entry)
    entry.insert(3, 0)
    entry[-1] -= 1 #todo: entry -2 ??

    if start_t > entry[1]:
        entry[3] = ceil(sample_rate * (start_t - entry[1]))

    if stop_t < entry[2]:
        entry[4] = floor(sample_rate * (stop_t - entry[2])) + entry[4]
    return WavReaderInfo(*entry)


def get_named_audio_file_range_info(entry, start_t=0, stop_t=0):

    file_start = IntegerTimestamp(entry.start_time_sec, entry.start_time_usec)
    file_stop = IntegerTimestamp(entry.end_time_sec, entry.end_time_usec)
    first_sample = 0
    final_sample = entry.num_samples-1
    # print(IntegerTimestamp(1552676300.009059))
    if start_t > file_start:
        # first_sample = ceil(sample_rate * (start_t - float(file_start)))
        first_sample = ceil(float(((file_start*-1)+start_t)*sample_rate)) # todo: HACK, the above produces rounding issues.  ceil() likely makes them not matter in most cases, but number inside is not otherwise the same, which may cause occasional issues which are hard to troubleshoot

    if stop_t < entry[2]:
        # final_sample = floor(sample_rate * (stop_t - file_stop)) + final_sample
        final_sample = floor(float(((file_stop*-1)+stop_t) *sample_rate))+ final_sample #todo: similar hack as above if statement
    return WavReaderInfo(filename=entry.filename, file_start_time=file_start, file_end_time=file_stop, start_position=first_sample, end_position=final_sample)

AudioTableEntry = namedtuple('AudioTableInfo', ['filename', 'start_time_sec', 'start_time_usec', 'end_time_sec', 'end_time_usec', 'num_samples', 'sample_rate'])

def audio_file_info_generator(db_file, start_t, stop_t):
    table = 'audio'
    entries = get_db_entries(db_file, table, start_t, stop_t)

    file_info_gen = (
        get_named_audio_file_range_info(AudioTableEntry(*entry), start_t, stop_t)
        for entry in entries
    )

    return file_info_gen


def audio_file_info_list(db_file, start_t, stop_t):
    table = 'audio'
    entries = get_db_entries(db_file, table, start_t, stop_t)

    file_info = [
        get_named_audio_file_range_info(entry, start_t, stop_t)
        for entry in entries
    ]
