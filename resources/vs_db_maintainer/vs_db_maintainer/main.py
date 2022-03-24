#! .venv/bin/python3

import sys
# import argparse
import os
import glob
import json
from collections import namedtuple

from data_types import IntegerTimestamp

from .database_resources import GTI_db, Master_db, Audio_Query, WavIncompleteError
from .filenames import master_index_db_str, processed_files_json_str

AudioEntryMaster = namedtuple('AudioEntryMaster', ['filename', 'start_time_sec','start_time_usec', 'end_time_sec','end_time_usec', 'n_sample', 'sample_rate'])#**used**

gti = GTI_db()
master = Master_db()
audio = Audio_Query()

db_entries = dict(audio=[], orientation=[])

# processed_files = {}

def get_files_list(location, extension, exceptions=[]): 
    files = [
        os.path.basename(file)
        for file in sorted(glob.glob(os.path.join(location, '*' + extension)))
        if os.path.basename(file) not in exceptions
    ]
    return files


def get_files_generator(location, extension, exceptions=[]): #**used**

    files = (
        # os.path.basename(file)
        file
        for file in sorted(glob.glob(os.path.join(location, '*' + extension)))
        if file not in exceptions
    )

    return files


def update_audio_entries(working_file, last_time, last_n_samples):
    entries = audio.get_new_entries(os.path.join(files_loc, working_file), last_time, last_n_samples)
    if entries[2]:
        master.update_latest_audio(master_db_file, working_file, entries[1:-1])


# todo: right now, if unprocessed_files is single string, treats as iterable letter by letter
def add_new_audio_entries(unprocessed_files):
    if isinstance(unprocessed_files, str):
        entries = (unprocessed_files, *audio.get_new_entries(os.path.join(files_loc, unprocessed_files), 0, 0))
        # entries = AudioEntryMaster(filename=unprocessed_files, *audio.get_new_entries(os.path.join(files_loc, unprocessed_files), 0, 0))
    else:
        # entries = ((working_file, *audio.get_new_entries(os.path.join(files_loc, working_file), 0, 0))
        #            for working_file in unprocessed_files)
        pass
    master.add_multiple_audio_entries(master_db_file, entries)

def update_orientation_entries(working_file, last_time):
    # working_file = last_orientation_file
    audio_id = master.get_audio_id(master_db_file, working_file)
    entries = gti.get_new_entries(os.path.join(files_loc, working_file), last_time)
    new_entries = ((working_file, audio_id, *entry) for entry in entries)
    if new_entries:
        master.add_multiple_orientation_entries(master_db_file, new_entries)


# todo: right now, if unprocessed_files is single string, treats as iterable letter by letter
def add_new_orientation_entries(unprocessed_files):
    if isinstance(unprocessed_files, str):
        working_file = unprocessed_files
        audio_id = master.get_audio_id(master_db_file, working_file)
        entries = gti.get_new_entries(os.path.join(files_loc, working_file), 0)
        start_time_sec = entries[0][0]
        start_time_usec = entries[0][1]
        end_time_sec = entries[-1][0]#todo:will need to revisit what to do when file is incomplete.
        end_time_usec = entries[-1][1]
        num_samples = len(entries)
        # new_entries = ((working_file, audio_id, *entry) for entry in entries)
        new_entries = (working_file, audio_id, start_time_sec, start_time_usec, end_time_sec, end_time_usec, num_samples)
        if new_entries:
            master.add_single_orientation_entry(master_db_file, new_entries)
    else:
        # for working_file in unprocessed_files:
        #     audio_id = master.get_audio_id(master_db_file, working_file)
        #     entries = gti.get_new_entries(os.path.join(files_loc, working_file), 0)
        #     new_entries = ((working_file, audio_id, *entry) for entry in entries)
        #     if new_entries:
        #         master.add_multiple_orientation_entries(master_db_file, new_entries)
        pass
OrientationEntryMaster = namedtuple('OrientationEntryMaster', ['filename', 'start_time_sec','start_time_usec','end_time_sec','end_time_usec','num_samples'])
def get_new_orientation_entry(unprocessed_files,):

    working_file = unprocessed_files
    # audio_id = master.get_audio_id(master_db_file, working_file)
    entries = gti.get_new_entries(os.path.join(files_loc, working_file), 0)
    start_time_sec = entries[0][0]
    start_time_usec = entries[0][1]
    end_time_sec = entries[-1][0]#todo:will need to revisit what to do when file is incomplete.
    end_time_usec = entries[-1][1]
    num_samples = len(entries)
    # new_entries = ((working_file, audio_id, *entry) for entry in entries)
    new_entries = OrientationEntryMaster(working_file,start_time_sec, start_time_usec, end_time_sec, end_time_usec, num_samples)

    return new_entries

        # for working_file in unprocessed_files:
        #     audio_id = master.get_audio_id(master_db_file, working_file)
        #     entries = gti.get_new_entries(os.path.join(files_loc, working_file), 0)
        #     new_entries = ((working_file, audio_id, *entry) for entry in entries)
        #     if new_entries:
        #         master.add_multiple_orientation_entries(master_db_file, new_entries)


def parse_inputs(data_location, index_location):#**used**
    global master_db_file
    global files_loc
    global db_loc

    # todo: wtf should I do with the input args
    # files_loc = os.path.abspath(sys.argv[1])
    # master_db_file = os.path.abspath(sys.argv[2])
    # db_loc = os.path.split(master_db_file)[0]
    files_loc = data_location
    if index_location:
        master_db_file = os.path.join(index_location, master_index_db_str)
    else:
        master_db_file = os.path.join(files_loc, master_index_db_str)
    db_loc = os.path.split(master_db_file)[0]


def load_processed_files_json(location, file):#**used**
    with open(os.path.join(location, file)) as f:
        return json.load(f)


def dump_processed_files_json(location, file, data):
    with open(os.path.join(location, file), 'w') as f:
        json.dump(data, f)


def connected(current_start, prev_end, threshold=10):#**used**
    gap = abs(current_start - prev_end)
    if (gap > float(threshold)):
        return False
    else:
        return True

def dump_processed_files():
    master.add_multiple_audio_entries(master_db_file, db_entries['audio'])
    master.add_multiple_orientation_entries(master_db_file, db_entries['orientation'])
    db_entries['audio'] = []
    db_entries['orientation'] = []
    dump_processed_files_json(db_loc, processed_files_json_str, processed_files) # Problem with this function. Somehow it is erasign the json file. Perhaps this has to do with the shadowing error?

def update(data_location, index_location = ''):#*****
    parse_inputs(data_location, index_location)
    global processed_files
    processed_files = load_processed_files_json(db_loc, processed_files_json_str)

    # all_db_files = get_files_generator(files_loc, '.db', exceptions=processed_files['orientation'])
    all_wav_files = get_files_generator(files_loc, '.wav', exceptions=processed_files['audio'])

    i = 0
    while True:  # update with iterable
        try:
            wav_file = next(all_wav_files)
            root_file = wav_file.split('.')[0]
            db_file = root_file + '.db'
            hdr_file = root_file + '.hdr'

            # print(wav_file)
            # add_single_new_audio_entry(wav_file)
            hdr_time = IntegerTimestamp(audio.read_hdr(
                os.path.join(files_loc, hdr_file))) # todo: can I incorporate better?
            current_time = IntegerTimestamp(**processed_files['current_time'])
            # print(i)
            if connected(hdr_time, current_time):  # todo: add a connected (1,0) field
                start_time = current_time + 1 / processed_files['sample_rate']
                file_data = audio.get_named_new_entries(os.path.join(files_loc, wav_file),
                                                        start_time=start_time)  # specify start time
            else:
                file_data = audio.get_named_new_entries(os.path.join(files_loc, wav_file),
                                                        start_time=hdr_time)  # specify start time
            # entries = AudioEntryMaster(wav_file, *file_data)
            current_audio_entry = AudioEntryMaster(wav_file, *file_data)
            # entries = (wav_file, *audio.get_named_new_entries(os.path.join(files_loc, wav_file)))
            # master.add_single_audio_entry(master_db_file, current_audio_entry)
            db_entries['audio'].append(current_audio_entry)
            processed_files['audio'].append(wav_file)
            processed_files['current_time'] = IntegerTimestamp(current_audio_entry.end_time_sec, current_audio_entry.end_time_usec).toJSON()  # todo check on time, maybe incorrect

            # db_file = next(all_db_files)
            # add_new_orientation_entries(db_file)
            current_orientation_entry = get_new_orientation_entry(db_file)
            db_entries['orientation'].append(current_orientation_entry)
            processed_files['orientation'].append(db_file)
            i += 1

            if (i >= 100):
                # master.add_multiple_audio_entries(master_db_file,db_entries['audio'])
                # master.add_multiple_orientation_entries(master_db_file, db_entries['orientation'])
                # db_entries['audio'] = []
                # db_entries['orientation']=[]
                # dump_processed_files_json(db_loc, 'processed_files.json', processed_files)
                dump_processed_files()
                print('{} processed'.format(wav_file))
                i = 0


        except WavIncompleteError:
            print('Need to do something with incomplete files')
        # except IndexError as err:
        #     dump_processed_files_json(db_loc, 'processed_files.json', processed_files)
        #     print('File {} has undetermined problem. Aborting...'.format(db_file))
        #     break

        except StopIteration:
            print('done')
            break

    dump_processed_files()

if __name__ == '__main__':
    update()
