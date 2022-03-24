import tensorflow as tf
import scipy.signal as scipy_signal
from scipy.signal import lfilter
import numpy as np
import datetime
import signal
import os

SAMPLE_RATE = 4000
# These calcuations only need to be made once for all spectrograms
# used to convert time hparam to number of sample points for tf.signal.stft
# 0.001 converts msecs to seconds
WIN_SIZE = (int)((250 * .001) * 4000 )
OVERLAP = 75
# converts a overlap percent into number of sample points
STEP = (int)((75 /100) * WIN_SIZE)
SAMPLE_POINTS = 1024
# mel bins is only used in MFCCs
MEL_BINS = 128
# MFCC Parameters
# lower bound frequency in hz, selected to not include lower frequency DC signal
LOWER_BOUND = 10.0
# upper bound frequency in hz, selected to be max possible frequency
UPPER_BOUND = 2000.0

SEG_HAPPENED = False

# Calibrates a segment from a wavcrawler object
def wavcrawler_calibration_filter(segment):
    #Channel 1
    #a_p=[1,0.2220]
    #b_p=[390.9326,390.9326]
    a_p =[1.000000000000000, 0.222030940703315]
    #b_p =[3.909326208402378, 3.909326208402378]*100 # Does not multiply each element in list
    b_p = [390.93262084, 390.93262084] # Already multiplied
    channel1 = lfilter(b_p, a_p, segment.samples[0])

    #Channel 2
    #a_x =[1.0000, -1.5363, 0.3679, 0.1684]
    #b_x = [0.0006, -0.0014, 0.0012, -0.0004]
    a_x = [1.000000000000000, -1.536296332383263, 0.367924214578871 , 0.168372117804392]
    b_x = [0.000568576018648, -0.001420271029412 , 0.001227373520296 , -0.000366496498110]
    channel2 = lfilter(b_x, a_x, segment.samples[1])

    #Channel 3
    #a_y = [1.0000, -1.5363, 0.3679, 0.1684]
    #b_y = [0.0006, -0.0014, 0.0012, -0.0004]
    a_y = [1.000000000000000, -1.536296332383263, 0.367924214578871, 0.168372117804392]
    b_y = [0.000568865591359, -0.001420994365816, 0.001227998615035, -0.000366683152807]
    channel3 = lfilter(b_y, a_y, segment.samples[2])

    #Channel 4
    #b_z = [.0010,0.5008,-0.7677,0.5969,-0.2848]
    #a_z = [1.0000, -1.1020, 0.0301, 0.0720]
    a_z = [1.000000000000000, -1.102026219228359, 0.030075503167086, 0.071950716061274]
    #b_z = [0.500806962684010, -0.767745850102613, 0.596854760271160, -0.284758287858497]/1000 # Does not divide each element in list
    b_z = [0.00050081, -0.00076775,  0.00059685, -0.00028476] # already divided
    channel4 = lfilter(b_z, a_z, segment.samples[3])

    calibrated_segment = np.array([channel1, channel2, channel3, channel4])

    return calibrated_segment

# Calibrates an entire wavcrawler object if just calibrated data is desired, not tensorflow or mel spectrogram
def batch_calibration(segment):
    calibrated_dataset = []
    for slice in segment:
        calibrated_segment = wavcrawler_calibration_filter(segment)
        calibrated_dataset.append(calibrated_segment)

    calibrated_dataset = np.array(calibrated_dataset)

    return calibrated_dataset

# Times are return as a list of lists, the first list is the start time, second is the end time
# segment length is in seconds
def wavcrawler_data_process(segment, mode, channels, segment_dur, calibrate=False):
    #print("Processing wavcrawler data and converting to tf data")
    target_num_samples = segment_dur * SAMPLE_RATE
    # For a single prediction, don't pass the entire wavcrawler object, just pass the segment
    # ensure the segment is 30 seconds, doesn't support different time lengths
    if mode == 'single':
        times=[]
        for time in segment[2]:
            timestamp_indicator = 1
            for item in time:
                if timestamp_indicator==1:
                    times.append(item)
                    timestamp_indicator = 0
                elif timestamp_indicator==0:
                    timestamp_indicator=1
        if channels == 1:
            if calibrate:
                calibrated_segment = wavcrawler_calibration_filter(segment)
                #downsample = scipy_signal.resample(calibrated_segment[0], 120000)
                downsample = scipy_signal.resample(calibrated_segment[0], target_num_samples)
            else:
                #downsample = scipy_signal.resample(segment[0][0], 120000)
                downsample = scipy_signal.resample(segment[0][0], target_num_samples)

            audio = tf.convert_to_tensor([downsample], np.float32)
            #audio = tf.reshape(audio, [120000,1])
            #audio = tf.reshape(audio, [target_num_samples,1])
            audio = tf.transpose(audio)
        elif channels == 4:
            temp_data = []
            if calibrate:
                calibrated_segment = wavcrawler_calibration_filter(segment)
                for i in range(0,4):
                    #data = scipy_signal.resample(calibrated_segment[i], 120000)
                    data = scipy_signal.resample(calibrated_segment[i], target_num_samples)
                    #np.append(test, data, axis=0)
                    temp_data.append(data)
            else:
                for i in range(0,4):
                    #data = scipy_signal.resample(segment[0][i], 120000)
                    data = scipy_signal.resample(segment[0][i], target_num_samples)
                    #np.append(test, data, axis=0)
                    temp_data.append(data)

            audio = np.array(temp_data)
            audio = tf.convert_to_tensor(audio, np.float32)
            #audio = tf.reshape(audio, [120000, 4])
            #audio = tf.reshape(audio, [target_num_samples, 4])
            audio = tf.transpose(audio)

    # For a batch, pass the entire wavcrawler object and this will iterate through the segments
    # Ensure each segment is 30 seconds
    elif mode == 'batch':
        audio = []
        times = []
        if channels == 1:
            for slice in segment:
                # save start time and end time to a list
                tmp_time = []
                for time in slice[2]:
                    timestamp_indicator = 1
                    for item in time:
                        if timestamp_indicator==1:
                            tmp_time.append(item)
                            timestamp_indicator = 0
                        elif timestamp_indicator==0:
                            timestamp_indicator=1
                times.append(tmp_time)
                # end saving start and end times
                if calibrate:
                    calibrated_slice = wavcrawler_calibration_filter(slice)
                    #downsample = scipy_signal.resample(calibrated_slice[0], 120000)
                    downsample = scipy_signal.resample(calibrated_slice[0], target_num_samples)
                else:
                    #downsample = scipy_signal.resample(slice[0][0], 120000)
                    downsample = scipy_signal.resample(slice[0][0], target_num_samples)

                audio_slice = tf.convert_to_tensor([downsample], np.float32)
                #audio_slice = tf.reshape(audio_slice, [120000,1])
                #audio_slice = tf.reshape(audio_slice, [target_num_samples,1])
                audio_slice = tf.transpose(audio_slice)
                audio_slice = audio_slice.numpy()
                audio.append(audio_slice)
        elif channels == 4:
            for slice in segment:
                # save start time and end time to a list
                tmp_time = []
                for time in slice[2]:
                    timestamp_indicator = 1
                    for item in time:
                        if timestamp_indicator==1:
                            tmp_time.append(item)
                            timestamp_indicator = 0
                        elif timestamp_indicator==0:
                            timestamp_indicator=1
                times.append(tmp_time)
                # end saving start and end times

                temp_data = []

                if calibrate:
                    calibrated_slice = wavcrawler_calibration_filter(slice)
                    for i in range(0,4):
                        #data = scipy_signal.resample(calibrated_slice[i], 120000)
                        data = scipy_signal.resample(calibrated_slice[i], target_num_samples)
                        #np.append(test, data, axis=0)
                        temp_data.append(data)
                else:
                    for i in range(0,4):
                        #data = scipy_signal.resample(slice[0][i], 120000)
                        data = scipy_signal.resample(slice[0][i], target_num_samples)
                        #np.append(test, data, axis=0)
                        temp_data.append(data)

                audio_slice = np.array(temp_data)
                audio_slice = tf.convert_to_tensor(audio_slice, np.float32)
                #audio_slice = tf.reshape(audio_slice, [120000, 4])
                #audio_slice = tf.reshape(audio_slice, [target_num_samples, 4])
                audio_slice = tf.transpose(audio_slice)
                audio_slice = audio_slice.numpy()
                audio.append(audio_slice)
        times = np.array(times)
        #print(times)
        times = times.T
        #print(times)
        #print("Finish coverting wavcrawler data")

    return audio, times

def generate_single_channel_mfcc(audio, mode, source, segment_dur):
    """
        This function processes the filepaths read-in from the csv file into mel log STFTs
    """        
    #duration=30

    if source == 'wav':
        audio_data = tf.io.read_file(audio)
        audio, _ = tf.audio.decode_wav(audio_data, desired_channels=1)
    
    # tf.squeeze(audio) to change shape to just samples, removes number of channels
    audio_squeeze = tf.reshape(tf.squeeze(audio), [1,-1])
    stfts = tf.signal.stft(audio_squeeze, frame_length=WIN_SIZE, frame_step=STEP, fft_length=SAMPLE_POINTS, window_fn=tf.signal.hann_window, pad_end=True )
    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = (SAMPLE_POINTS //2) +1

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(MEL_BINS, num_spectrogram_bins, SAMPLE_RATE, LOWER_BOUND, UPPER_BOUND)
    mel_spectrograms = tf.tensordot( spectrograms, linear_to_mel_weight_matrix, 1)

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    # Compute MFCCs from log_mel_spectrograms
    #mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)
    # calculate the time axis for conversion
    time_space = (int)(segment_dur * SAMPLE_RATE) // STEP
    # mfcc function returns channels, time, freq, need to convert to time, freq, channels for CNNs
    final_mfcc =  tf.reshape(tf.squeeze(log_mel_spectrograms), [time_space, MEL_BINS, 1])
    
    #if self.repeat:
        #final_mfcc = tf.repeat(final_mfcc, repeats=3, axis=2)

    if mode=='single':
        final_mfcc = tf.data.Dataset.from_tensor_slices([final_mfcc])

    return final_mfcc

def generate_multi_channel_mfcc(audio, mode, source):
    """
        This function processes the filepaths read-in from the csv file into mel log STFTs
        This function exists only to process multi-channel files 
    """        

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = (SAMPLE_POINTS //2) +1

    if source == 'wav':
        audio_data = tf.io.read_file(audio)
        audio, _ = tf.audio.decode_wav(audio_data, desired_channels=4)

    #print("Splitting data")
    channels = tf.split(audio, num_or_size_splits=4, axis=1) 
    # tf.squeeze(audio) to change shape to just samples, removes number of channels
    all_channels = []
    #print("iterating through channels")
    for ch in channels:
        audio_squeeze = tf.reshape(tf.squeeze(ch), [1,-1])
        stfts = tf.signal.stft(audio_squeeze, frame_length=WIN_SIZE, frame_step=STEP, fft_length=SAMPLE_POINTS, window_fn=tf.signal.hann_window, pad_end=True )
        spectrograms = tf.abs(stfts)

        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(MEL_BINS, num_spectrogram_bins, SAMPLE_RATE, LOWER_BOUND, UPPER_BOUND)
        mel_spectrograms = tf.tensordot( spectrograms, linear_to_mel_weight_matrix, 1)

        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
        all_channels.append(tf.squeeze(log_mel_spectrograms))
    # mfcc function returns channels, time, freq, need to convert to time, freq, channels for CNNs
    final_mfcc =  tf.stack([all_channels[0], all_channels[1], all_channels[2], all_channels[3]], axis=2)

    if mode=='single':
            final_mfcc = tf.data.Dataset.from_tensor_slices([final_mfcc])

    return final_mfcc

def sig_handler(signum, frame):
    SEG_HAPPENED = True
    print("Segfault occurred")
    print("Trying again")

# Combines the above functions to take filepath as input, output data for making prediction
# Works on either a single 30 seconds, or a batch of 30 minutes (makes predictions on each 30 second period)
# data can either be list of filepaths or segments from the wavcrawler object
# channels is the desired number of channels for the data to make a prediction
# mode is single (for a single 30 second period) or batch (for a batch of 30 minutes of data)
# source is whether the source are .wav filepaths (wav) or segments from the wavcrawler (wc)
# batch size is the number of 30 second segments (each 30 second segment is an individual prediction)
# segment length is the number of seconds per segment, should match what the model was trained with
# Calibrate is whether or not to calibrate the data (True, False)
def full_mel_mfcc_pipeline(data, channels, mode, source, segment_dur, calibrate):

    #signal.signal(signal.SIGSEGV, sig_handler)

    if source == 'wc':
        data, times = wavcrawler_data_process(data, mode, channels, segment_dur, calibrate)

    if mode == 'single':
        if channels==1:
            #spectrogram = generate_spectrogram_mfcc(filepaths)
            #dataset = generate_mel_spectrogram_mfcc(spectrogram)
            dataset = generate_single_channel_mfcc(data, mode, source, segment_dur)
        elif channels==4:
            dataset = generate_multi_channel_mfcc(data, mode, source)
        dataset = dataset.batch(1)

    elif mode == 'batch':
        batch_size = len(data)
        
        dataset = tf.data.Dataset.from_tensor_slices(data)
        
        #seg=True
        #while seg:
        #    SEG_HAPPENED = False
        if channels==1:
        # Perform spectrogram operation in single function
            print("Performing map function (single channel)")
            dataset = dataset.map(lambda x: generate_single_channel_mfcc(x, mode, source, segment_dur), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        elif channels==4:
            print("Performing map function (four channel)")
            #print(data)
            #print(len(data))
            dataset = dataset.map(lambda x: generate_multi_channel_mfcc(x, mode, source), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #if SEG_HAPPENED:
        #    continue
        #else:
        #    seg=False
        # The map function returns the dataset for segments as a dataset of tensors (each tensor is 30 seconds)
        # However, when the map function works on filenames, it is returned as a dataset of a single tensor, with
        # each 30 second chunk as a list. This code converts the dataset of tensors to the dataset of a single
        # tensor, but it needs to be revisited because it may not be efficient even though it works
        # This won't work on large datasets because it stores everything in memory (all_segs)
        # Retry by using wavcrawler object, converting each segment individually and returning as numpy array

        print("Dataset processed")
        dataset = dataset.batch(batch_size)
    else:
        print('Please select either single or batch for data processing mode')

    # Change batch size if doing more than 1 30 second file
    
    return dataset, times

def prediction_db_write(predictions, engine):
    predictions['record_timestamp'] = datetime.datetime.utcnow() #(datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds()
    predictions.to_sql(name='predictions', con=engine, index=False, if_exists='append')

'''

# This function creates the spectrograms that are used to plot the graphs
def generate_spectrogram_mfcc(filepath):
    audio_data = tf.io.read_file(filepath)
    #audio_data = tf.io.read_file("/home/lemgog/thesis/acoustic-inference-application/data/single_testdata/classA_130321_16_60.wav")
    #audio_data = tf.io.read_file("/h/nicholas.villemez/thesis/acoustic-inference-application/data/single_testdata/classA_130321_16_60.wav")

    # Gives 2d array (number of samples (sample rate * time (times number of channels if relevant)), and 1)
    audio, _ = tf.audio.decode_wav(audio_data, desired_channels=1)

    # tf.squeeze(audio) to change shape to just samples, removes number of channels
    # Just get number of samples
    audio_squeeze = tf.reshape(tf.squeeze(audio), [1,-1])

    # Gives 3d array, number of channels, time, frequency
    # frame length is window length and number of samples. Input in scripts is in milliseconds time.
    # Multiple time domain by sample rate to get time amount we want. 
    # 1 second at 4000khz, is 4000 samples, 
    # Frame step, number of samples to step. Discrete fourier transforms, have to know how far to move each time
    # Input to scripts is 75 overlap, is a percentage. Convert 75% to number, percent of frame length
    # If frame length 1000, 75% overlap is 750 in frame step
    # fft_length: integer, size of fourier transform to apply. Powers of 2.
    # Window function, takes window length, hann window is shape of window function in each step
    # Pad end: pad end of signals with zeroes when frame length and step is past end
    # Returns multi-dim array (channels, frames, fft unique bins (fft length/2 + 1))
    # time = duration * sample rate / step size. Step is overlap
    # 30 second, 4000 * 30 (seconds) = 120,000, 
    stfts = tf.signal.stft(audio_squeeze, frame_length=WIN_SIZE, frame_step=STEP, fft_length=SAMPLE_POINTS, window_fn=tf.signal.hann_window, pad_end=True )
    spectrograms = tf.abs(stfts)

    return spectrograms

# This function continues from the spectrogram function and makes mel spectrograms for the model to make predictions on
def generate_mel_spectrogram_mfcc(spectrograms):

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = (SAMPLE_POINTS //2) +1

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(MEL_BINS, num_spectrogram_bins, SAMPLE_RATE, LOWER_BOUND, UPPER_BOUND)
    mel_spectrograms = tf.tensordot( spectrograms, linear_to_mel_weight_matrix, 1)

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    # Compute MFCCs from log_mel_spectrograms
    #mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)
    # calculate the time axis for conversion
    time_space = (int)(30 * 4000) // STEP
    # mfcc function returns channels, time, freq, need to convert to time, freq, channels for CNNs
    final_mfcc =  tf.reshape(tf.squeeze(log_mel_spectrograms), [time_space, MEL_BINS, 1])

    repeat = False
    if repeat:
        final_mfcc = tf.repeat(final_mfcc, repeats=3, axis=2)

    #dataset = dataset.cache() 

    dataset = tf.data.Dataset.from_tensor_slices([final_mfcc])

    return dataset

# This function creates the spectrograms that are used to plot the graphs
# filepaths is a list of filepaths
def generate_spectrogram_mfcc_30minbatch(filepaths, channels):

    # Make tensor dataset of filepaths
    dataset = tf.data.Dataset.from_tensor_slices(filepaths)

    dataset = dataset.map(lambda x: generate_spectrogram_mfcc(x, channels), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset

# This function continues from the spectrogram function and makes mel spectrograms for the model to make predictions on
def generate_mel_spectrogram_mfcc_30minbatch(spectrograms):

    dataset = spectrograms.map(lambda x: generate_mel_spectrogram_mfcc(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset

# Function to perform all options if desired
# Not currently used
def acoustic_data_pipeline(filepath, channels):

    if channels == 1:
        SAMPLE_RATE = 4000
        # These calcuations only need to be made once for all spectrograms
        # used to convert time hparam to number of sample points for tf.signal.stft
        # 0.001 converts msecs to seconds
        WIN_SIZE = (int)((250 * .001) * 4000 )
        OVERLAP = 75
        # converts a overlap percent into number of sample points
        STEP = (int)((75 /100) * WIN_SIZE)
        SAMPLE_POINTS = 1024
        # mel bins is only used in MFCCs
        MEL_BINS = 128


        # MFCC Parameters
        # lower bound frequency in hz, selected to not include lower frequency DC signal
        LOWER_BOUND = 10.0
        # upper bound frequency in hz, selected to be max possible frequency
        UPPER_BOUND = 2000.0

        audio_data = tf.io.read_file(filepath)
        #audio_data = tf.io.read_file("/home/lemgog/thesis/acoustic-inference-application/data/single_testdata/classA_130321_16_60.wav")
        #audio_data = tf.io.read_file("/h/nicholas.villemez/thesis/acoustic-inference-application/data/single_testdata/classA_130321_16_60.wav")

        # Gives 2d array (number of samples (sample rate * time (times number of channels if relevant)), and 1)
        audio, _ = tf.audio.decode_wav(audio_data, desired_channels=1)

        # tf.squeeze(audio) to change shape to just samples, removes number of channels
        # Just get number of samples
        audio_squeeze = tf.reshape(tf.squeeze(audio), [1,-1])

        # Gives 3d array, number of channels, time, frequency
        # frame length is window length and number of samples. Input in scripts is in milliseconds time.
        # Multiple time domain by sample rate to get time amount we want. 
        # 1 second at 4000khz, is 4000 samples, 
        # Frame step, number of samples to step. Discrete fourier transforms, have to know how far to move each time
        # Input to scripts is 75 overlap, is a percentage. Convert 75% to number, percent of frame length
        # If frame length 1000, 75% overlap is 750 in frame step
        # fft_length: integer, size of fourier transform to apply. Powers of 2.
        # Window function, takes window length, hann window is shape of window function in each step
        # Pad end: pad end of signals with zeroes when frame length and step is past end
        # Returns multi-dim array (channels, frames, fft unique bins (fft length/2 + 1))
        # time = duration * sample rate / step size. Step is overlap
        # 30 second, 4000 * 30 (seconds) = 120,000, 
        stfts = tf.signal.stft(audio_squeeze, frame_length=WIN_SIZE, frame_step=STEP, fft_length=SAMPLE_POINTS, window_fn=tf.signal.hann_window, pad_end=True )
        spectrograms = tf.abs(stfts)

        # Warp the linear scale spectrograms into the mel-scale.
        num_spectrogram_bins = (SAMPLE_POINTS //2) +1

        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(MEL_BINS, num_spectrogram_bins, SAMPLE_RATE, LOWER_BOUND, UPPER_BOUND)
        mel_spectrograms = tf.tensordot( spectrograms, linear_to_mel_weight_matrix, 1)

        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

        # Compute MFCCs from log_mel_spectrograms
        #mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)
        # calculate the time axis for conversion
        time_space = (int)(30 * 4000) // STEP
        # mfcc function returns channels, time, freq, need to convert to time, freq, channels for CNNs
        final_mfcc =  tf.reshape(tf.squeeze(log_mel_spectrograms), [time_space, MEL_BINS, 1])

        repeat = False
        if repeat:
            final_mfcc = tf.repeat(final_mfcc, repeats=3, axis=2)

        #dataset = dataset.cache() 

        dataset = tf.data.Dataset.from_tensor_slices([final_mfcc])
        dataset = dataset.batch(1)


        #acoustic_inference_model = tf.keras.models.load_model('/home/lemgog/thesis/acoustic-inference-application/models/1-channel-checkpoint-0.75.h5')
        #acoustic_inference_model = tf.keras.models.load_model('/h/nicholas.villemez/thesis/acoustic-inference-application/models/1-channel-checkpoint-0.75.h5')
        
        #acoustic_inference_model = tf.keras.models.load_model(model)
        #predict_probs = acoustic_inference_model.predict(dataset)

        #predict_labels = predict_probs.argmax(axis=-1)

    elif channels == 4:
        pass

    #prediction_dictionary = {0:'Class A', 1:'Class B', 2:'Class C', 3:'Class D', 4:'Class E'}
    #predict_labels = [prediction_dictionary[x] for x in predict_labels]

    #return predict_labels
    return dataset
'''