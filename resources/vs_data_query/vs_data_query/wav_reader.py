import aubio
#todo: figure out what to do with examples

sample_rate = 8000
n_channels = 4

from collections import namedtuple
from math import floor, ceil
from numpy import hstack

from .helper_functions import * #todo: update with function names to clarify namespace

# todo: end of time frame?

class WavSegmentReader():
    def __init__(self, filename, samplerate=sample_rate, hop_size=512, channels=n_channels, position=0, overlap=0):
        self.source = aubio.source(filename, samplerate=samplerate, hop_size=hop_size, channels=channels)
        self.source.seek(position)
        self.read = 0
        self.position = position
        self.hop_size = hop_size
        self.overlap = overlap
        if not 0 <= overlap <= 1:
            raise ValueError('Overlap must be in range 0 to 1')

    def __iter__(self):
        return self

    def __next__(self):
        next_position = self.get_next_position()
        return self.read_next_segment(next_position=next_position)

    def read_next_segment(self, next_position):
        samples = next(self.source)
        self.read += samples.shape[1]
        self.position = next_position
        self.source.seek(self.position)
        return samples

    def get_next_position(self):
        return self.position + int(self.hop_size - self.hop_size * self.overlap)


SegmentData = namedtuple('SegmentData', ['samples', 'time_stamp','time_range', 'position_range','sample_rate'])


class WavCrawler():
    def __init__(self, db_file, min_t=None, max_t=None, samplerate=8000, segment_length=512, channels=4, overlap=0):
        self.files_gen = audio_file_info_generator(db_file, min_t, max_t)
        self.time_range = dict(min_t=min_t, max_t=max_t)
        self.current_info = next(self.files_gen)
        reader_info = dict(filename=self.current_info.filename, samplerate=samplerate, hop_size=segment_length,
                           position=self.current_info.start_position, channels=channels, overlap=overlap)
        self.segment_reader = WavSegmentReader(**reader_info)
        self.i = 0
        self._tmp_file_info = None

    def __iter__(self):
        return self

    def __next__(self):
        return self.handle_next_segment()  # todo: possibly add window function here?

    def handle_next_segment(self):
        try:
            return self.get_simple_segment()
        except EndOfSeriesError:
            raise StopIteration #do this if we are done with processing
        except EndOfFileError:
            return self.get_overlapping_segment()

    def get_simple_segment(self):
        time_range = (self.current_info.file_start_time + self.segment_reader.position / sample_rate,
                      self.current_info.file_start_time + self.segment_reader.position / sample_rate + self.segment_reader.hop_size / sample_rate)

        position_range = (self.segment_reader.position, self.segment_reader.position + self.segment_reader.hop_size)
        if time_range[1] > self.time_range['max_t']:
            raise EndOfSeriesError
        if time_range[1] >= self.current_info.file_end_time:
            raise EndOfFileError
        time_stamp = time_range[0] + (time_range[1] - time_range[0])/2
        samples = next(self.segment_reader)
        return SegmentData(samples=samples, time_stamp=time_stamp,time_range=time_range, position_range=position_range, sample_rate=sample_rate)


    def get_overlapping_segment(self):
        time_range0 = (self.current_info.file_start_time + self.segment_reader.position / sample_rate,
                       self.current_info.file_end_time)

        pos0 = self.segment_reader.position - self.current_info.end_position
        samples0 = next(self.segment_reader.source)

        if self._tmp_file_info:
            next_file_info = self._tmp_file_info
        else:
            next_file_info = next(self.files_gen)

        tmp_segment_length = self.segment_reader.hop_size - samples0.shape[1]

        next_reader_info = dict(filename=next_file_info.filename, samplerate=sample_rate, hop_size=tmp_segment_length,
                            position=0, channels=n_channels, overlap=0)

        pos1 = tmp_segment_length
        if tmp_segment_length: #tmp_hop_size is 0 when last file read has read perfectly up to last sample of file.
            tmp_reader = WavSegmentReader(**next_reader_info)
            samples1 = next(tmp_reader)
            time_range1 = (next_file_info.file_start_time + tmp_reader.position / sample_rate,
                           next_file_info.file_start_time + (tmp_reader.hop_size - 1) / sample_rate)
            samples = hstack((samples0, samples1))
            time_range = (time_range0[0], time_range1[1])

        else: #this block deals with the edge case, where the preceding file has been read up to its last sample, with no overlap into next file
            samples = samples0
            time_range = time_range0
        position_range = (pos0, pos1)

        next_position = pos1 - int(self.segment_reader.hop_size * self.segment_reader.overlap)
        if next_position < 0:
            self.segment_reader.position = self.current_info.end_position + next_position
            self.segment_reader.source.seek(self.segment_reader.position)
            self._tmp_file_info = next_file_info

        else:
            self.current_info = next_file_info
            reader_info = dict(filename=next_file_info.filename, samplerate=sample_rate,
                               hop_size=self.segment_reader.hop_size,
                               position=next_position, channels=n_channels, overlap=self.segment_reader.overlap)
            self.segment_reader = WavSegmentReader(**reader_info)
            self._tmp_file_info = None
        time_stamp = time_range[0] + (time_range[1] - time_range[0]) / 2
        return SegmentData(samples=samples, time_stamp=time_stamp, time_range=time_range,position_range=position_range, sample_rate=sample_rate)


class EndOfFileError(ValueError):
    pass


class EndOfSeriesError(ValueError):
    pass
