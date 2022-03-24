# This script is to redo all of the predictions in the database

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

def prediction_db_update(predictions, start_time, end_time, connection):
    insert_statement = "UPDATE predictions SET model_predictions='" + predictions + "' WHERE start_time=" + str(start_time) + " AND end_time=" + str(end_time)
    connection.execute(insert_statement)

    #predictions['record_timestamp'] = datetime.datetime.utcnow() #(datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds()
    #predictions.to_sql(name='predictions', con=engine, index=False, if_exists='append')

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

print("Establish database connections...")
engine = db.create_engine(SQLALCHEMY_DATABASE_URI)
connection = engine.connect()

# Get first prediction time from prediction database

#pred_query = 'SELECT start_time, end_time FROM predictions'

#prediction_times = pd.read_sql_query(pred_query, connection)

# Iterate through times, set t1 and t2

first_predict_time_query = 'SELECT end_time FROM predictions where id=(SELECT MAX(id) FROM predictions)' #'SELECT MIN(start_time) FROM predictions' 
first_predict_time = connection.execute(first_predict_time_query)
first_time = first_predict_time.all()[0][0]


latest_predict_time_query = 'SELECT MAX(end_time) FROM predictions'
latest_predict_time = connection.execute(latest_predict_time_query)
last_time = latest_predict_time.all()[0][0]

t1 = first_time

print("Querying models from database")

query = 'SELECT * FROM Models WHERE channels = 1 AND active=True'


single_channel_models = pd.read_sql_query(query, engine)


query = 'SELECT * FROM Models WHERE channels = 4 AND active=True'
four_channel_models = pd.read_sql_query(query, engine)

num_models = four_channel_models.shape[0] + single_channel_models.shape[0]

while t1<last_time:
    t2 = t1 + 1800

    # Start timer to record how long predictions took
    
    print("Making predictions for " + str(datetime.datetime.utcfromtimestamp(t1)) + " until " + str(datetime.datetime.utcfromtimestamp(t2)))
    
    #-----------------------------------------------------------------------------------
    # If stop iteration reached, then querying data that does not exist yet
    desired_seconds = 30
    segment_length = desired_seconds * WC_SAMPLE_RATE
    try:
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
        four_channel_dataset, timestamps = full_mel_mfcc_pipeline(four_channel_wc, channels=4, mode='batch', source='wc', segment_dur=desired_seconds, calibrate=False)
        single_channel_dataset, timestamps = full_mel_mfcc_pipeline(single_channel_wc, channels=1, mode='batch', source='wc', segment_dur=desired_seconds, calibrate=False)
    except (StopIteration, ValueError) as e:
        print("No new acoustic data, data processing pipeline")
        continue

    '''
    print("t1: " + str(t1))
    print("t2: " + str(t2))

    #pred_query = 'SELECT start_time, end_time FROM predictions WHERE start_time>=' + str(t1) + ' AND end_time<=' + str(t2)

    #prediction_times = pd.read_sql_query(pred_query, connection)

    for times in prediction_times.iterrows():
        print(times['start_time'])
    
    print(timestamps[0])
    for times in prediction_times.iterrows():
        print(times['end_time'])
    print(timestamps[1])
    
    db_start_times = []
    db_end_times = []
    for i in range(len(timestamps[0])):
        start_time = timestamps[0][i]
        end_time = timestamps[1][i]

        pred_query = 'SELECT start_time, end_time FROM predictions WHERE start_time=' + str(start_time) + ' AND end_time=' + str(end_time)
        new_times = connection.execute(pred_query)
        try:
            time_tup = new_times.all()[0]
        except IndexError:
            print("Index Error")
            print(start_time)
            print(end_time)
        new_start_time = time_tup[0]
        new_end_time = time_tup[1]
        print(new_start_time)
        print(new_end_time)
        db_start_times.append(new_start_time)
        db_end_times.append(new_end_time)

    print(timestamps[0])
    print(db_start_times)
    print(timestamps[1])
    print(db_end_times)
    #print(prediction_times)
    #print(timestamps)
    input("Make sure timestamps match")

    '''

    if not single_channel_models.empty:
        single_channel_model_files = single_channel_models[['file_name', 'id']].copy()
        single_channel_model_files.loc[:,'file_name'] = single_channel_model_files.apply(lambda x: MODEL_DIRECTORY + "/id_" + str(x['id']) + '/' + x['file_name'], axis=1)
        # Convert params for models to json format
        single_channel_model_files['params'] = single_channel_models['params'].apply(lambda x: json.loads(x)) 

    

    if not four_channel_models.empty:
        four_channel_model_files = four_channel_models[['file_name', 'id']].copy()
        four_channel_model_files.loc[:,'file_name'] = four_channel_model_files.apply(lambda x: MODEL_DIRECTORY + "/id_" + str(x['id']) + '/' + x['file_name'], axis=1)
        four_channel_model_files['params'] = four_channel_models['params'].apply(lambda x: json.loads(x))

    # Check if there are any models, if there are none, then continue
    if four_channel_models.empty and single_channel_models.empty:
        print("No models active for predictions, waiting for 5 minutes. Please upload models for inference")
        time.sleep(300)
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



    # update model predictions for each time period in database
    print("Storing predictions in database")

    #prediction_df.apply(lambda x: prediction_db_update(x['model_predictions'], x['start_time'], x['end_time'], connection), axis=1)

    try:
        prediction_db_write(prediction_df, engine)
    except psycopg2.errors.StringDataRightTruncation as e:
        print("Size error saving predictions to database, like too many models in database and size of predictions column is too big, please delete one")
        print("Error: " + e)    

    t1 = t2


'''
you can drop id at and below this after new predictions made

inference_db=# SELECT MAX(id) FROM predictions;
  max  
-------
 52817

 '''


acoustic_connection.close()
acoustic_engine.dispose()

connection.close()
engine.dispose()