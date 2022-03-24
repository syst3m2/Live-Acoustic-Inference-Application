#
# Created by paulleary on 11/13/19
#
import wave as wav
import datetime

from collections import namedtuple

from data_types import IntegerTimestamp
from sqlite_resources import create_connection, execute_sql_command, execute_multiple_sql_command


#todo: need to do something about runs that don't add to json file.
class Master_db():
    def __init__(self):
        self.tables = ['orientation', 'audio']
        orientation_elements = ['time_sec', 'time_usec', 'roll_degrees', 'pitch_degrees', 'yaw_degrees'] # NOT USED
        audio_elements = ['filename', 'time_sec', 'time_usec'] # NOT USED
        self.table_elements = {self.tables[0]: orientation_elements, self.tables[1]: audio_elements} # NOT USED

    def create_tables(self, db_file):
        conn = create_connection(db_file)

        # create_audio_table_str = """ CREATE TABLE IF NOT EXISTS audio (
        #                                                 id integer PRIMARY KEY,
        #                                                 filename text NOT NULL,
        #                                                 start_time_sec integer NOT NULL,
        #                                                 start_time_usec integer NOT NULL,
        #                                                 end_time_sec integer NOT NULL,
        #                                                 end_time_usec integer NOT NULL,
        #                                                 num_samples integer NOT NULL,
        #                                                 sample_rate integer NOT NULL
        #                                             ); """
        #
        # create_orientation_table_str = """CREATE TABLE IF NOT EXISTS orientation (
        #                                     id integer PRIMARY KEY,
        #                                     filename text NOT NULL,
        #                                     audio_id integer NOT NULL,
        #                                     start_time_sec integer NOT NULL,
        #                                     start_time_usec integer NOT NULL,
        #                                     end_time_sec integer NOT NULL,
        #                                     end_time_usec integer NOT NULL,
        #                                     num_samples integer NOT NULL,
        #                                     FOREIGN KEY (audio_id) REFERENCES audio (id)
        #                                 );"""
        create_audio_table_str = """ CREATE TABLE IF NOT EXISTS audio (
                                                                filename text NOT NULL,
                                                                start_time_sec integer NOT NULL,
                                                                start_time_usec integer NOT NULL,
                                                                end_time_sec integer NOT NULL,
                                                                end_time_usec integer NOT NULL,
                                                                num_samples integer NOT NULL, 
                                                                sample_rate integer NOT NULL                                                 
                                                            ); """

        create_orientation_table_str = """CREATE TABLE IF NOT EXISTS orientation (
                                                    filename text NOT NULL,
                                                    start_time_sec integer NOT NULL,
                                                    start_time_usec integer NOT NULL,
                                                    end_time_sec integer NOT NULL,
                                                    end_time_usec integer NOT NULL, 
                                                    num_samples integer NOT NULL      
                                                );"""
        with conn:
            execute_sql_command(conn, create_audio_table_str)
            execute_sql_command(conn, create_orientation_table_str)

    def get_audio_latest(self, db_file):
        conn = create_connection(db_file)
        with conn:
            cmd = """SELECT end_time, num_samples, filename
                                                FROM audio
                                                WHERE end_time = (
                                                SELECT MAX(end_time)
                                                FROM audio);"""

            cur = execute_sql_command(conn, cmd)
            entries = cur.fetchall()
            assert len(entries) <= 1, 'Master has more than 1 maximum, recommend using time_usec'
            out = [[], [], []]
            if (entries):
                entry = entries[0]
                out = [entry[0],
                       entry[1],
                       entry[2]]
            return out

    def get_missing_audio_files(self, db_file, files_list):
        # todo: fix string formatting here
        val_str = "('{}')".format("'),('".join(files_list))
        conn = create_connection(db_file)
        with conn:
            cmd = "CREATE TEMPORARY TABLE IF NOT EXISTS files(id integer PRIMARY KEY,name text NOT NULL)"
            cur = conn.execute(cmd)
            cmd = " INSERT INTO files(name) VALUES {}".format(val_str)
            cur = conn.execute(cmd)
            cmd = """SELECT name
                        FROM files
                        LEFT JOIN audio t
                        ON files.name = t.filename
                        WHERE t.id IS NULL;"""
            cur = conn.execute(cmd)
            entries = cur.fetchall()
            missing_files = [e[0] for e in entries]
            return (missing_files)

    def add_multiple_audio_entries(self, db_file, entries):

        # cmd = """ INSERT INTO audio(filename,start_time,end_time,num_samples,sample_rate)
        #               VALUES(?,?,?,?,?)"""

        cmd = """ INSERT INTO audio(filename,start_time_sec,start_time_usec,end_time_sec, end_time_usec,num_samples,sample_rate)
                              VALUES(?,?,?,?,?,?,?)"""

        conn = create_connection(db_file)
        with conn:
            execute_multiple_sql_command(conn, cmd, entries)

    def add_single_audio_entry(self, db_file, entry):#**used**

        cmd = """ INSERT INTO audio(filename,start_time_sec,start_time_usec,end_time_sec, end_time_usec,num_samples,sample_rate)
                      VALUES(?,?,?,?,?,?,?)"""

        conn = create_connection(db_file)
        with conn:
            execute_sql_command(conn, cmd, entry)

    def update_latest_audio(self, db_file, filename, entries):
        cmd = """UPDATE audio
                    SET end_time = ? ,
                        num_samples = ? 
                    WHERE filename = ? """
        entries = (*entries, filename)
        conn = create_connection(db_file)
        with conn:
            execute_sql_command(conn, cmd, entries)

    def get_audio_id(self, db_file, working_file):#**used**
        audio_file = working_file.split('.')[0] + '.wav'
        entry = (audio_file,)
        cmd = """SELECT id FROM audio WHERE filename = ?"""
        conn = create_connection(db_file)
        with conn:
            cur = execute_sql_command(conn, cmd, entry)
            id = cur.fetchall()[0][0]
        return id

    def get_orientation_latest(self, db_file):
        conn = create_connection(db_file)
        with conn:
            cmd = """SELECT time_sec, time_usec, filename
                        FROM orientation
                        WHERE time_sec = (
                        SELECT MAX(time_sec)
                        FROM orientation);"""
            cur = execute_sql_command(conn, cmd)
            entries = cur.fetchall()
            assert len(entries) <= 1, 'Master has more than 1 maximum, recommend looking into this'
            out = [[], []]
            if (entries):
                entry = entries[0]
                out = [entry[0],
                       entry[2]]
            return out

    def get_missing_orientation_files(self, db_file, files_list):
        # todo: fix string formatting here
        val_str = "('{}')".format("'),('".join(files_list))
        conn = create_connection(db_file)
        with conn:
            cmd = "CREATE TEMPORARY TABLE IF NOT EXISTS files(id integer PRIMARY KEY,name text NOT NULL)"
            cur = conn.execute(cmd)
            cmd = " INSERT INTO files(name) VALUES {}".format(val_str)
            cur = conn.execute(cmd)
            cmd = """SELECT name
                        FROM files
                        LEFT JOIN orientation t
                        ON files.name = t.filename
                        WHERE t.id IS NULL;"""
            cur = conn.execute(cmd)
            entries = cur.fetchall()
            missing_files = [e[0] for e in entries]
            return missing_files

    def add_multiple_orientation_entries(self, db_file, entries):
        # cmd = """ INSERT INTO orientation(filename,audio_id,time_sec,time_usec,roll_degrees,pitch_degrees, yaw_degrees)
        #               VALUES(?,?,?,?,?,?,?)"""
        cmd = """ INSERT INTO orientation(filename,start_time_sec,start_time_usec,end_time_sec, end_time_usec,num_samples)
                              VALUES(?,?,?,?,?,?)"""
        conn = create_connection(db_file)
        with conn:
            execute_multiple_sql_command(conn, cmd, entries)

    def add_single_orientation_entry(self, db_file, entries):#**used**
        # cmd = """ INSERT INTO orientation(filename,audio_id,time_sec,time_usec,roll_degrees,pitch_degrees, yaw_degrees)
        #               VALUES(?,?,?,?,?,?,?)"""
        cmd = """ INSERT INTO orientation(filename,start_time_sec,start_time_usec,end_time_sec, end_time_usec,num_samples)
                              VALUES(?,?,?,?,?,?)"""
        conn = create_connection(db_file)
        with conn:
            execute_sql_command(conn, cmd, entries)


class GTI_db():
    def __init__(self):
        self.table = 'orientation_sensor_d3de1b93_d581_4c2c_8238_a12c780f8c81'
        self.elements = ['time_sec', 'time_usec', 'roll_degrees', 'pitch_degrees', 'yaw_degrees']

    def get_new_entries(self, db_file, previous_time): #**used**
        conn = create_connection(db_file)
        entry = (previous_time,)
        cmd = """SELECT time_sec, time_usec, roll_degrees, pitch_degrees, yaw_degrees
                      FROM orientation_sensor_d3de1b93_d581_4c2c_8238_a12c780f8c81 WHERE time_sec>?"""

        with conn:
            cur = execute_sql_command(conn, cmd, entry)
            entries = cur.fetchall()
            return entries

    def get_old_entries(self, db_file, previous_time):
        conn = create_connection(db_file)
        entry = (previous_time,)
        cmd = """SELECT time_sec, time_usec, roll_degrees, pitch_degrees, yaw_degrees
              FROM orientation_sensor_d3de1b93_d581_4c2c_8238_a12c780f8c81 WHERE time_sec<=?"""

        with conn:
            cur = execute_sql_command(conn, cmd, entry)
            entries = cur.fetchall()
            return entries


# todo: perhaps this can be moved into the class
AudioEntries = namedtuple('AudioEntries', ['start_time_sec','start_time_usec', 'end_time_sec','end_time_usec', 'n_sample', 'sample_rate'])

class Audio_Query():
    def __init__(self):
        pass

    # def get_new_entries(self, wav_file, start_time=0, previous_time=0, previous_n_samples=0):
    #     entries = [None] * 4
    #     [n_samples, sample_rate] = self.get_wav_sampledata(wav_file)
    #     # [start_time, end_time, n_samples, sample_rate] = 0, 0, 0, 0
    #     if start_time == 0:
    #         # if we don't supply a start time, it is read from the hdr file
    #         hdr_file = wav_file.split('.')[0] + '.hdr'
    #         start_time = self.read_hdr(hdr_file)

    #     if previous_n_samples == 0:
    #         entries[0] = start_time

    #     if n_samples > previous_n_samples:
    #         diff_samples = n_samples - previous_n_samples
    #         end_time = start_time + previous_time + diff_samples / sample_rate
    #         entries[1:] = [end_time, n_samples, sample_rate]

    #     entries = tuple(entries)
    #     return entries

    def get_named_new_entries(self, wav_file, start_time=0, previous_time=0, previous_n_samples=0):  #**used**
        entries = [None] * 6
        [n_samples, sample_rate] = self.get_wav_sampledata(wav_file)
        # [start_time, end_time, n_samples, sample_rate] = 0, 0, 0, 0
        if start_time == 0:
            # if we don't supply a start time, it is read from the hdr file
            hdr_file = wav_file.split('.')[0] + '.hdr'
            start_time = IntegerTimestamp(self.read_hdr(hdr_file))

        if previous_n_samples == 0:
            entries[0:2] = [*start_time]

        if n_samples > previous_n_samples:
            diff_samples = n_samples - previous_n_samples
            end_time = start_time + previous_time + diff_samples / sample_rate # todo: is this the problem??
            entries[2:] = [*end_time, n_samples, sample_rate]

        # entries = tuple(entries)
        entries = AudioEntries(*entries)
        return entries

    def read_hdr(self, hdr_file):#**used**
        with open(hdr_file) as fid:
            contents = fid.read()
        t_str = ' '.join(contents.split()[8:11]) + ' +0000' #todo: added UTC offset as only way to force datetime to recognize timestamp as UTC.
        dt = datetime.datetime.strptime(t_str, '%j %H:%M:%S.%f %Y %z')
        return dt.timestamp()

    def get_wav_sampledata(self, file):#**used**
        try:
            with wav.open(file) as wr:
                n = wr.getnframes()
                f = wr.getframerate()
                out = [n, f]
                return out
        except wav.Error:
            print('file: {} possibly incomplete'.format(
                file))  # handle exception further, create dummy function to read another way
            raise WavIncompleteError


class WavIncompleteError(wav.Error):
    pass


if __name__ == '__main__':
    pass
