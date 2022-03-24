import pandas as pd
import sqlalchemy as db
from vs_data_query import WavCrawler
from acoustic_streamer import *
from predict import model_predict
from acoustic_streamer import prediction_db_write
import time
import os
import json
import sys
import datetime
import psycopg2

# Set/get environment variables
# Add functionality to save entropy and std to predictions
MODEL_DIRECTORY = os.environ.get('MODEL_FOLDER') #or '/home/lemgog/thesis/acoustic_app/model_predictor/models'

ACOUSTIC_DATABASE = os.environ.get('ACOUSTIC_DATABASE_URL') #or '/home/lemgog/thesis/acoustic_app/data/mbari/master_index.db'

SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') #or 'sqlite:///../acoustic-inference-application/app.db'

WC_SAMPLE_RATE = 8000

# Use to check the range for AIS true labels
def check_range(t1, t2, radius, mmsi, connection):
    query = 'SELECT * FROM AIS \
        WHERE "timeOfFix" BETWEEN ' + str(t1) + ' AND ' + str(t2) + \
        ' AND dist_from_sensor_km >' + str(radius) + ' AND mmsi=' + str(mmsi)

    check_range = pd.read_sql_query(query, connection)

    if check_range.empty:
        return True
    else:
        return False

def get_class(mmsi, connection):
    #print("mmsi: " + str(mmsi))
    query = "SELECT desig FROM mmsi WHERE mmsi=" + str(mmsi)
    desig = connection.execute(query).first()[0]
    class_query = "SELECT ship_class from ship_classes WHERE desig='" + desig +"'"
    ship_class = connection.execute(class_query).first()[0]
    #print("desig: " + desig)
    #print("ship class: " + ship_class)
    return ship_class

# Returns a JSON of true labels for a given time period
def get_true_labels(t1, t2, radius, connection):
    # t1 is 24 hours before the start time and t2 is desired time period after t1
    # which is 30 seconds in this case
    # Radius to evaluate predictions on is 30 kilometers
    label_t1 = int(t1) - (60*60*24)
    label_t2 = t2
    #radius = 20

    # Select the latest time where the radius is in the defined prediction radius within last 24 hours
    # until t2
    label_query = 'SELECT mmsi, MAX("timeOfFix") FROM AIS \
    WHERE "timeOfFix" BETWEEN ' + str(label_t1) + ' AND ' + str(label_t2) + ' AND dist_from_sensor_km <=' + str(radius) \
    + ' GROUP BY mmsi'

    true_labels = pd.read_sql_query(label_query, connection)

    # Need to fix this code to right json format
    # Maybe do inside get_class function if mmsi is empty?

    if not true_labels.empty:
        true_labels['ship_class'] = true_labels.apply(lambda x: get_class(x['mmsi'], connection), axis=1)

        # perform another query between latest time for that ship and t1, if ship left radius before prediction, then not a true label
        true_labels = true_labels[true_labels.apply(lambda x: check_range(x['max'], t1, radius, x['mmsi'], connection), axis=1)]['ship_class']
        
        true_labels.name = "True Labels"

        true_labels = true_labels.to_json(orient='records')

        if len(json.loads(true_labels)) == 0:
            true_labels = '["Class E"]'

    elif true_labels.empty:
        true_labels = '["Class E"]'

    #print(true_labels)
    return true_labels

def get_all_labels(t1, t2, radius_list, connection):
    #print("T1: " + str(t1))
    #print("T1: " + str(datetime.datetime.fromtimestamp(int(t1))))
    #print("T2: " + str(t2))
    #print("T2: " + str(datetime.datetime.fromtimestamp(int(t2))))
    label_dict = {}
    for radius in radius_list:
        labels = get_true_labels(t1, t2, radius, connection)
        label_dict[radius] = labels
    return json.dumps(label_dict)



#START_FLAG = True

while True:

    print("Establish database connections...")
    engine = db.create_engine(SQLALCHEMY_DATABASE_URI)
    connection = engine.connect()

    acoustic_engine = db.create_engine(ACOUSTIC_DATABASE)
    acoustic_connection = acoustic_engine.connect()
    print("database connections established")

    # Grab last predictions, compute next 30 minute chunk of data, perform predictions on it.
    # If file does not exist, then query database for the last prediction

    # If you want to run repeatedly on a sample dataset where there are already predictions, then select dev
    # If you want to continue from last set of predictions, then choose prod
    # If you want to start from the most recently saved audio data, then do prod-first
    # If you want to open up the container and run tests, select test
    '''
    if START_FLAG:
        if sys.argv[1] == 'dev-subset':
            print("Dev mode, Determining time range to make inferences on")
            t1_query = 'SELECT MIN(start_time_sec) FROM audio'
            t1 = acoustic_connection.execute(t1_query)
            t1 = t1.all()[0][0]
            #t1 = 1625460855
            #t1 = t1-1800
            START_FLAG = False
            t2 = t1 + 1800

        elif sys.argv[1] == 'dev-first':
            print("Prod first mode, Determining time range to make inferences on")
            latest_time_query = "SELECT MAX(end_time_sec) FROM AUDIO"
            t2 = acoustic_connection.execute(latest_time_query)
            t2 = t2.all()[0][0]
            START_FLAG = False
            t1 = t2 - 1800

        elif sys.argv[1] == 'dev-cont':
            print("Prod first mode, Determining time range to make inferences on")
            latest_time_query = "SELECT MAX(end_time_sec) FROM AUDIO"
            t2 = acoustic_connection.execute(latest_time_query)
            t2 = t2.all()[0][0]
            START_FLAG = False
            t1 = t2 - 1800

        elif sys.argv[1] == 'prod-cont':
            print("Prod mode, Determining time range to make inferences on")
            t1_query = 'SELECT MAX(end_time) FROM predictions'
            t1 = connection.execute(t1_query)
            t1 = t1.all()[0][0]
            START_FLAG = False
            t2 = t1 + 1800
            
        elif sys.argv[1] == 'prod-first':
            print("Prod first mode, Determining time range to make inferences on")
            latest_time_query = "SELECT MAX(end_time_sec) FROM AUDIO"
            t2 = acoustic_connection.execute(latest_time_query)
            t2 = t2.all()[0][0]
            START_FLAG = False
            t1 = t2 - 1800
            
        elif sys.argv[1] == 'test':
            print("In test mode, sleeping 10 minutes")
            time.sleep(600)
            continue
        
    
    # After the first run through, continuously check for the latest time from
    # audio database, grab latest time from predictions
    else:
    '''
    latest_audio_time_query = "SELECT MAX(end_time_sec) FROM AUDIO"
    latest_audio_time = acoustic_connection.execute(latest_audio_time_query)
    latest_audio_time = latest_audio_time.all()[0][0]

    latest_predict_time_query = 'SELECT MAX(end_time) FROM predictions'
    latest_predict_time = connection.execute(latest_predict_time_query)
    latest_predict_time = latest_predict_time.all()[0][0]

    # If the difference between the latest time from the audio database and the 
    # last prediction time is less than 30 minutes (1800 seconds), 
    # then there is not enough data in the database, wait 15 minutes, continue loop again
    # If there is enough data, define new t1 and t2 to make predictions for
    if latest_predict_time is not None:
        if abs(latest_audio_time - latest_predict_time) < 1800:
            print("Not enough data in database, waiting 15 minutes...")
            print("Latest predict time: " + str(datetime.datetime.utcfromtimestamp(latest_predict_time)))
            print("Latest audio time: " + str(datetime.datetime.utcfromtimestamp(latest_audio_time)))
            time.sleep(900)
            continue
        else:
            t1 = latest_predict_time
            t2 = t1 + 1800
    else:
        print("No predictions in the database, using audio database to initialize prediction timeline as starting point")
        t2 = latest_audio_time
        t1 = t2 - 1800
    # End else statement

    # Start timer to record how long predictions took
    start = time.perf_counter()
    print("Making predictions for " + str(datetime.datetime.utcfromtimestamp(t1)) + " until " + str(datetime.datetime.utcfromtimestamp(t2)))
    
    print("Querying models from database")
    query = 'SELECT * FROM Models WHERE channels = 1 AND active=True'

    try:
        single_channel_models = pd.read_sql_query(query, engine)
    except psycopg2.errors.UndefinedTable as e:
        print('Encountered Undefined Table Error while querying models table, table may not exist yet')
        print(e)
    except db.exc.ProgrammingError as e:
        print('Encountered error while querying models table')
        print(e)
    

    
    if not single_channel_models.empty:
        print('Single Channel Models Present')
        single_channel_model_files = single_channel_models[['file_name', 'id']].copy()
        single_channel_model_files.loc[:,'file_name'] = single_channel_model_files.apply(lambda x: MODEL_DIRECTORY + "/id_" + str(x['id']) + '/' + x['file_name'], axis=1)
        # Convert params for models to json format
        single_channel_model_files['params'] = single_channel_models['params'].apply(lambda x: json.loads(x)) 

    query = 'SELECT * FROM Models WHERE channels = 4 AND active=True'
    four_channel_models = pd.read_sql_query(query, engine)

    if not four_channel_models.empty:
        print('Four Channel Models Present')
        four_channel_model_files = four_channel_models[['file_name', 'id']].copy()
        four_channel_model_files.loc[:,'file_name'] = four_channel_model_files.apply(lambda x: MODEL_DIRECTORY + "/id_" + str(x['id']) + '/' + x['file_name'], axis=1)
        four_channel_model_files['params'] = four_channel_models['params'].apply(lambda x: json.loads(x))

    # Check if there are any models, if there are none, then continue
    if four_channel_models.empty and single_channel_models.empty:
        print("No models active for predictions, waiting for 5 minutes. Please upload models for inference")
        time.sleep(300)
        continue
 
    
    num_models = four_channel_models.shape[0] + single_channel_models.shape[0]

    #print("Making predictions for the following models: ")
    #print(four_channel_models[['file_name', 'id']])
    #print(single_channel_models[['file_name', 'id']])

    # Difference between t1 and t2 must be a multiple of 30 (predictions performed on 30 second increments)
    # need to store master_index.db in an os environment variable
    '''

    ---------NEED TO DEVELOP DYNAMIC METHOD TO GENERATE TIMES FOR PREDICTIONS----------
    ---------Need to check for new models in database each time make prediction-------

    '''

    
    #-----------------------------------------------------------------------------------
    # If stop iteration reached, then querying data that does not exist yet
    desired_seconds = 30
    segment_length = desired_seconds * WC_SAMPLE_RATE
    print("Extracting audio data")
    try:
        print("Wavcrawler try statement")
        #four_channel_wc = WavCrawler(ACOUSTIC_DATABASE[10:], t1, t2, segment_length=240000, overlap=0.25)
        #single_channel_wc = WavCrawler(ACOUSTIC_DATABASE[10:], t1, t2, segment_length=240000, overlap=0.25)
        four_channel_wc = WavCrawler(ACOUSTIC_DATABASE[10:], t1, t2, segment_length=segment_length, overlap=0.25)
        single_channel_wc = WavCrawler(ACOUSTIC_DATABASE[10:], t1, t2, segment_length=segment_length, overlap=0.25)
    except StopIteration:
        print("No new acoustic data, first exception")
        continue

    # Determine how many 30 second segments are in the wavcrawler to pass as batch size

    # Timestamps will be the same, so variable name doesn't matter
    # If stop iteration reached, then not a full 30 minutes worth of data
    try:
        print("Running data processing pipeline")
        print("Creating four channel dataset")
        four_channel_dataset, timestamps = full_mel_mfcc_pipeline(four_channel_wc, channels=4, mode='batch', source='wc', segment_dur=desired_seconds, calibrate=False)
        print("Creating single channel dataset")
        single_channel_dataset, timestamps = full_mel_mfcc_pipeline(single_channel_wc, channels=1, mode='batch', source='wc', segment_dur=desired_seconds, calibrate=False)
    except (StopIteration, ValueError) as e:
        print("No new acoustic data, data processing pipeline")
        continue


    # Move this logic inside the mfcc pipeline to return a list of start times and end times
    # Multiplies the list of times by the number of models to apply to each prediction later
    print("Creating timestamps")
    #print("These are the timestamps")
    #print(timestamps)
    if not single_channel_models.empty:
        single_start_times = [timestamps[0] for i in range(single_channel_model_files.shape[0])]
        single_end_times = [timestamps[1] for i in range(single_channel_model_files.shape[0])]
    #print("Single Start times and end times")
    #print(single_start_times)
    #print(single_end_times)
    if not four_channel_models.empty:
        four_start_times = [timestamps[0] for i in range(four_channel_model_files.shape[0])]
        four_end_times = [timestamps[1] for i in range(four_channel_model_files.shape[0])]

    # Flatten lists
    if not single_channel_models.empty:
        single_start_times = [times for sublist in single_start_times for times in sublist]
        single_end_times = [times for sublist in single_end_times for times in sublist]

    if not four_channel_models.empty:
        four_start_times = [times for sublist in four_start_times for times in sublist]
        four_end_times = [times for sublist in four_end_times for times in sublist]

    # Perform predictions on data for each model
    print("Making predictions")
    if not single_channel_models.empty:
        single_channel_model_files['predictions'] = single_channel_model_files.apply(lambda x: model_predict(x['params'], single_channel_dataset, x['file_name']), axis=1)

    if not four_channel_models.empty:
        four_channel_model_files['predictions'] = four_channel_model_files.apply(lambda x: model_predict(x['params'], four_channel_dataset, x['file_name']), axis=1)

    #four_channel_model_files.to_csv('uncertainty.csv', index=False)
    print("formatting predictions")
    # Need to fix in order to use Multilabel and add Entropy, STD to predictions
    # Drop params column from this, not needed

    # Prediction column is column of lists, this unpacks so each row corresponds to a prediction on a single 30 second period

    # test another way
    #print(single_channel_model_files['predictions'])
    if not single_channel_models.empty:
        single_model_df = pd.DataFrame()

        for index, row in single_channel_model_files.iterrows():
            #pred_df = pd.DataFrame(json.loads(row['predictions']))

            pred_df = pd.DataFrame(row['predictions'])
            # if columns include entropy and std, combine into one column comma separated
            #print(pred_df)
            #print(pred_df.shape[1])
            if pred_df.shape[1]>2:
                pred_df[0] = pred_df[0].astype(str) + ',' + pred_df[1].astype(str) + ',' + pred_df[2].astype(str)
                pred_df=pred_df.drop(2, axis=1)
                pred_df=pred_df.drop(1, axis=1)
            pred_df['id'] = row['id']
            #print(pred_df)
            single_model_df = pd.concat([single_model_df, pred_df])

        single_model_df.rename(columns={0:'predictions'}, inplace=True)
        single_channel_model_files = single_model_df

    

    if not four_channel_models.empty:
        four_model_df = pd.DataFrame()
        for index, row in four_channel_model_files.iterrows():
            #pred_df = pd.DataFrame(json.loads(row['predictions']))
            pred_df = pd.DataFrame(row['predictions'])
            # if columns include entropy and std, combine into one column comma separated
            if pred_df.shape[1]>2:
                pred_df[0] = pred_df[0].astype(str) + ',' + pred_df[1].astype(str) + ',' + pred_df[2].astype(str)
                pred_df=pred_df.drop(2, axis=1)
                pred_df=pred_df.drop(1, axis=1)
            pred_df['id'] = row['id']
            four_model_df = pd.concat([four_model_df, pred_df])

        four_model_df.rename(columns={0:'predictions'}, inplace=True)
        four_channel_model_files = four_model_df

    #four_channel_model_files.to_csv('uncert_test.csv', index=False)

    # add timestamps to each prediction
    if not single_channel_models.empty:
        #print(single_start_times)
        #print(single_channel_model_files['predictions'])
        #print(single_channel_model_files)
        single_channel_model_files['start_time'] = single_start_times
        single_channel_model_files['end_time'] = single_end_times

    if not four_channel_models.empty:
        four_channel_model_files['start_time'] = four_start_times
        four_channel_model_files['end_time'] = four_end_times

    # Combine single and four channel predictions
    # groupby start time and end time, combine model id's and predictions, format column as json

    if not single_channel_models.empty and not four_channel_model_files.empty:
        prediction_df = pd.concat([single_channel_model_files, four_channel_model_files])
    elif not single_channel_models.empty and four_channel_model_files.empty:
        prediction_df = single_channel_model_files
    elif single_channel_models.empty and not four_channel_model_files.empty:
        prediction_df = four_channel_model_files
    
    #prediction_df = prediction_df.astype('str') 
    #prediction_df['model_predictions'] = '"' + prediction_df['id'] + '"' + ':' + '"' + prediction_df['predictions'].apply(json.dumps) + '"'
    prediction_df['model_predictions'] = '"' + prediction_df['id'].astype('str') + '"' + ':' + prediction_df['predictions'].apply(json.dumps)

    prediction_df = prediction_df.groupby(['start_time', 'end_time'])['model_predictions'].apply(','.join).reset_index()
    prediction_df['model_predictions'] = '{' + prediction_df['model_predictions'] + '}'

    #prediction_df.to_csv('uncert_pred_test.csv', index=False)
    # Maybe get rid of!
    # Grab true labels for that time period

    #Make predictions without true labels
    '''
    print("Gathering true labels for time period")
    # Define a radius to get true labels for, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120
    radius_list = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    prediction_df['true_label'] = prediction_df.apply(lambda x: get_all_labels(x['start_time'], x['end_time'], radius_list, connection), axis=1)
    '''

    # store model predictions for each time period in database
    print("Storing predictions in database")
    try:
        prediction_db_write(prediction_df, engine)
    except psycopg2.errors.StringDataRightTruncation as e:
        print("Size error saving predictions to database, like too many models in database and size of predictions column is too big, please delete one")
        print("Error: " + e)


    stop = time.perf_counter()

    # Write time to make predictions on chunk of last predictions to file
    time_json = "{" + '"start_time":' + str(prediction_df['start_time'].min()) + ',' + \
                   '"end_time":' + str(prediction_df['end_time'].max()) + \
                    ', "predict_time(sec)":' + str((stop-start)) + ', "num_models":' + str(num_models) + "}" 
    timer = open('/home/tracking/predict_timer.json', 'a+')
    timer.write(time_json)
    timer.close()

    #Total time to make predictions
    print("Total time to make prediction on 30 minutes of data with " + str(num_models) + " models: " + str((stop-start)/60) + " minutes")

    # Warn if predict times will cause increasing time-latency
    if ((stop-start)/60) > 30:
        print("WARNING: PREDICT TIME TAKING TOO LONG (>30 minutes), WILL CAUSE INCREASING TIME LATENCY BEHIND ACOUSTIC STREAM")

    acoustic_connection.close()
    acoustic_engine.dispose()

    connection.close()
    engine.dispose()
    