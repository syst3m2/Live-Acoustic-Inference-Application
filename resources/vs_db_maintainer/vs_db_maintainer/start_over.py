from .database_resources import Master_db
from .filenames import master_index_db_str, processed_files_json_str
import os
import sys
import json

master = Master_db()

# loc = os.path.abspath('/Volumes/mars_ext/')
# db_file = os.path.join(loc, 'master.db')
# loc = os.path.abspath('/Volumes/marsdata/meta')
# master_db_file = os.path.join(loc, 'master_index.db')
# json_file = os.path.join(loc, 'processed_files.json')

def parse_inputs(location):
    global master_db_file
    global json_file
    # global files_loc
    # loc =  os.path.abspath(sys.argv[1])
    master_db_file = os.path.join(location, master_index_db_str)
    json_file = os.path.join(location, processed_files_json_str)
    #    files_loc = os.path.abspath(sys.argv[2])

def delete_create(db_file):
    try:
        os.remove(db_file)
    except FileNotFoundError:
        pass
    master.create_tables(db_file)

def new_json_file(file_path):
    status = {}
    status['audio'] = []
    status['orientation'] = []
    status['current_time'] = dict(sec=0, usec=0)
    status['sample_rate'] = 8000
    with open(file_path, 'w') as f:
        json.dump(status, f)

def main(location):
    parse_inputs(location)
    delete_create(master_db_file)
    new_json_file(json_file)

def refresh(location):
    main(location)

if __name__ == '__main__':
    refresh()
