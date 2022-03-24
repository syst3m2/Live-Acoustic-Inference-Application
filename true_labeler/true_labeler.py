import datetime
from pandas.io.parsers import CParserWrapper
import sqlalchemy as db
import pandas as pd
import os
import json
import sys
import time

radius_list = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

app_db = os.environ.get('DATABASE_URL')

'''
def combine_classes(t1, t2, mmsi_df):
    class_list = mmsi_df[(((mmsi_df['start_time']>=t1) & (mmsi_df['start_time']<=t2)) | ((mmsi_df['end_time']>=t1) & (mmsi_df['end_time']<=t2)) | ((mmsi_df['start_time']<=t1) & (mmsi_df['end_time']>=t2)))]['ship_class'].tolist()
    # If there are no ships in that time in the radius, then insert Class E
    if len(class_list)==0:
        class_list.append('Class E')

    return class_list
'''
'''
def combine_classes(t1, t2, mmsi_df, radius_list):
    class_list = []
    true_label_dict = {}
    for radius in radius_list:
        mmsi_radius_df = mmsi_df[mmsi_df['radius']==radius]
        class_list = mmsi_radius_df[(((mmsi_radius_df['start_time']>=t1) & (mmsi_radius_df['start_time']<=t2)) | \
                                    ((mmsi_radius_df['end_time']>=t1) & (mmsi_radius_df['end_time']<=t2)) | \
                                    ((mmsi_radius_df['start_time']<=t1) & (mmsi_radius_df['end_time']>=t2)) | \
                                    ((mmsi_radius_df['start_time']>=t1) & (mmsi_radius_df['end_time']<=t2)))]['ship_class'].tolist()
        # If there are no ships in that time in the radius, then insert Class E
        if len(class_list)==0:
            class_list=['Class E']

        true_label_dict[radius]=class_list

    true_label_dict = json.dumps(true_label_dict)

    return true_label_dict
'''
def combine_classes(t1, t2, mmsi_df, radius_list):
    class_list = []
    true_label_dict = {}
    for radius in radius_list:
        
        # If there are no AIS positions at all for the day, then ikely the AIS stream was broken, label as none
        if mmsi_df.empty:
            class_list=[None]

        else:
            if radius != max(radius_list):
                classe_radius = radius+10
            else:
                classe_radius = radius
            mmsi_radius_df = mmsi_df[mmsi_df['radius']==radius]
            mmsi_classe_df = mmsi_df[mmsi_df['radius']==classe_radius]

            class_list = mmsi_radius_df[(((mmsi_radius_df['start_time']>=t1) & (mmsi_radius_df['start_time']<=t2)) | \
                                        ((mmsi_radius_df['end_time']>=t1) & (mmsi_radius_df['end_time']<=t2)) | \
                                        ((mmsi_radius_df['start_time']<=t1) & (mmsi_radius_df['end_time']>=t2)) | \
                                        ((mmsi_radius_df['start_time']>=t1) & (mmsi_radius_df['end_time']<=t2)))]['ship_class'].tolist()
            # If there are no ships in that time in the radius, then insert Class E

            classe_list = mmsi_classe_df[(((mmsi_classe_df['start_time']>=t1) & (mmsi_classe_df['start_time']<=t2)) | \
                                        ((mmsi_classe_df['end_time']>=t1) & (mmsi_classe_df['end_time']<=t2)) | \
                                        ((mmsi_classe_df['start_time']<=t1) & (mmsi_classe_df['end_time']>=t2)) | \
                                        ((mmsi_classe_df['start_time']>=t1) & (mmsi_classe_df['end_time']<=t2)))]['ship_class'].tolist()
            
            if len(classe_list)==0:
                class_list=['Class E']
            elif len(class_list)==0 and len(classe_list)>0:
                class_list=[None]

        true_label_dict[radius]=class_list

    true_label_dict = json.dumps(true_label_dict)

    return true_label_dict


def insert_true_labels(true_labels,start_time,end_time, connection):
    insert_statement = "UPDATE predictions SET true_label='" + true_labels + "' WHERE start_time=" + str(start_time) + " AND end_time=" + str(end_time)
    app_connection.execute(insert_statement)

    return


# Pass in a dataframe of ais tracks and a radius, this will get the start/end times for each mmsi
def get_times(ais_df, previous_day_dict, radius, today_timestamp):
    mmsi_dict = {}
    next_day_dict = {}
    grouped_ais = ais_df.groupby('mmsi')
    enter_loop = False
    for mmsi, group in grouped_ais:
        # Make sure dataframe sorted by time so iterates through chronologically
        group = group.sort_values(by=['timeOfFix'])
        cpa = 0
        cpa_time = 0
        num_passes = 1

        # If the mmsi is in the previous day dict, 
        # add it to the mmsi dict as it's first entry
        
        for key in previous_day_dict:
            if key is not None:
                #print(previous_day_dict[key])
                if str(mmsi) in str(key['mmsi']) and str(mmsi) not in mmsi_dict:
                    if key['radius']==radius:
                        mmsi_dict[mmsi] = {num_passes:key}
        
        for index, row in group.iterrows():
            ship_class = row['ship_class']

            '''            
            if str(mmsi) in previous_day_dict and mmsi not in mmsi_dict:
                if previous_day_dict[mmsi]['radius']
                mmsi_dict[mmsi] = {num_passes:previous_day_dict[mmsi]}
            '''

            if row['ship_class'] == 'Unknown':
                print("Unknown ship class")
                break
            

            if row['dist_from_sensor_km'] <= radius and row['ship_class'] != "Unknown":
                enter_loop=True
                # This is the first time the mmsi enters the range
                if row['ship_class'] != "Unknown" and mmsi not in mmsi_dict:  
                    # else add new key to dict with only start time
                    # don't add unknown ships
                    mmsi_dict[mmsi] = {num_passes:{"mmsi":mmsi, "radius":radius, "start_time":row['timeOfFix'], "end_time":row['timeOfFix'], "cpa":row['dist_from_sensor_km'], "cpa_time":row['timeOfFix'], "desig":row['desig'], "ship_class":row['ship_class']}}
                    #mmsi_dict[radius][mmsi] = [mmsi, row['timeOfFix'], row['timeOfFix'], rng, row['timeOfFix'], row['desig'], row['ship_class']]        
                    #if brg_and_rng:
                    #daily_ships[mmsi].append([[record_time, brg, rng]])

                # This is for when the ship passes through the first time again after a previous pass
                elif row['ship_class'] != "Unknown" and mmsi in mmsi_dict and num_passes not in mmsi_dict[mmsi]:
                    mmsi_dict[mmsi][num_passes] = {"mmsi":mmsi, "radius":radius, "start_time":row['timeOfFix'], "end_time":row['timeOfFix'], "cpa":row['dist_from_sensor_km'], "cpa_time":row['timeOfFix'], "desig":row['desig'], "ship_class":row['ship_class']}

                # After the ship has entered and the first entry is in the dict, update the end time and CPA
                if mmsi in mmsi_dict and num_passes in mmsi_dict[mmsi]: #and num_passes==1:
                    # Update CPA if range is less than what's recorded
                    if row['dist_from_sensor_km'] < mmsi_dict[mmsi][num_passes]['cpa']:
                        
                        mmsi_dict[mmsi][num_passes]['end_time'] = row['timeOfFix']
                        # Update CPA
                        mmsi_dict[mmsi][num_passes]['cpa'] = row['dist_from_sensor_km']
                        # update CPA time
                        mmsi_dict[mmsi][num_passes]['cpa_time'] = row['timeOfFix']

                    # If the CPA is no longer close, then update the end time until ship is outside radius
                    else:
                        # Update end time, stop updating cpa
                        mmsi_dict[mmsi][num_passes]['end_time'] = row['timeOfFix']                

            #If the mmsi and the current pass are already in the dictionary and  the position is outside of range, 
            # that means the ship already transited through and just exited the loop
            # Iterate num_passes to create new entry for mmsi
            if mmsi in mmsi_dict and row['dist_from_sensor_km'] > radius and num_passes in mmsi_dict[mmsi]:
                enter_loop=False
                num_passes += 1

        # If reach the end of the group, and enter loop is still true, 
        # and the end time is less than 1 hour from the next day, then ship never exited
        # the radius, add to the next day tracker
        # Otherwise, assume latest time is the end time in range
        if enter_loop and mmsi in mmsi_dict:
            end_time = int(mmsi_dict[mmsi][num_passes]['end_time'])
            if today_timestamp - end_time < 3600:
                print(str(mmsi) + " Never exited, carrying over to next day")
                next_day_dict[mmsi] = mmsi_dict[mmsi][num_passes]

    # Convert nested dict to dataframe

    # There is a problem here and the dictionary is being saved down wrong
    mmsi_df = pd.DataFrame()

    for key in mmsi_dict:
        tmp_df = pd.DataFrame(mmsi_dict[key]).T
        mmsi_df = pd.concat([mmsi_df, tmp_df],ignore_index=True)

    next_day_df = pd.DataFrame(next_day_dict).T #.reset_index(drop=True)
    #next_day_df.to_json('next_day.json', orient='index')
    next_day_df = next_day_df.reset_index(drop=True)

    return mmsi_df, next_day_df

#START_FLAG = True
app_engine = db.create_engine(app_db)
app_connection = app_engine.connect()

while True:
    '''
    if START_FLAG:
        if sys.argv[1] == 'dev-subset':
            print("Dev mode, working with subset of data, dates need to be set explicitly in code")
            # Set to the date range of subset of data working with
            yesterday = datetime.date(2021,7,4)
            today = datetime.date(2021,7,6)
            last_label_day = datetime.date(2021,7,3)
            two_days = datetime.date(2021,7,3)
            yesterday_timestamp = (yesterday - datetime.date(1970, 1, 1)).total_seconds()
            today_timestamp = (today - datetime.date(1970, 1, 1)).total_seconds()
            last_true_label_timestamp = (last_label_day - datetime.date(1970, 1, 1)).total_seconds()

        elif sys.argv[1] == 'dev-first':
            print("dev first mode")
            print("Starting true labeler for first time, sleeping for 24 hours")
            time.sleep(86400)
            START_FLAG=False
            today = datetime.datetime.utcnow().date() 
            yesterday = today - datetime.timedelta(days=1)
            two_days = today - datetime.timedelta(days=2)
            tomorrow = datetime.datetime.combine(today,datetime.datetime.min.time()) + datetime.timedelta(days=1)
            last_label_day = two_days
            last_true_label_timestamp = (last_label_day - datetime.date(1970, 1, 1)).total_seconds()
            yesterday_timestamp = (yesterday - datetime.date(1970, 1, 1)).total_seconds()
            today_timestamp = (today - datetime.date(1970, 1, 1)).total_seconds()

        # Choose this if there are already predictions in the database that require true labels
        elif sys.argv[1] == 'dev-cont':
            print("dev mode")
            START_FLAG=False
            today = datetime.datetime.utcnow().date() 
            yesterday = today - datetime.timedelta(days=1)
            two_days = today - datetime.timedelta(days=2)
            tomorrow = datetime.datetime.combine(today,datetime.datetime.min.time()) + datetime.timedelta(days=1)
            #last_label_day = two_days
            #last_day_file = open("last_label_day.json","r")
            #last_label_day = json.loads(last_day_file.read())
            #last_label_day = datetime.datetime.strptime(last_label_day, "%Y-%m-%d").date()

            check_query = "SELECT MAX(start_time) FROM PREDICTIONS WHERE true_label IS NOT NULL"
            last_true_label = app_connection.execute(check_query).first()

            check_query2 = "SELECT MIN(start_time) FROM PREDICTIONS WHERE true_label IS NULL"
            first_empty_label = app_connection.execute(check_query2).first()

            if last_true_label is not None:
                last_true_label_timestamp = last_true_label[0]
                last_label_day = datetime.utcfromtimestamp(int(last_true_label_timestamp)).date()
            elif first_empty_label is not None:
                print("No predictions have true labels yet, using first NULL entry as starting point")
                last_true_label_timestamp = first_empty_label[0]
                last_label_day = datetime.utcfromtimestamp(int(last_true_label_timestamp)).date()
            else:
                print("No predictions in database with true labels to be filled in. \
                        Check predict stream")

            #last_true_label_timestamp = (last_label_day - datetime.date(1970, 1, 1)).total_seconds()
            yesterday_timestamp = (yesterday - datetime.date(1970, 1, 1)).total_seconds()
            today_timestamp = (today - datetime.date(1970, 1, 1)).total_seconds()

        elif sys.argv[1] == 'prod-first':
            print("Prod first mode")
            print("Starting true labeler for first time, sleeping for 24 hours")
            time.sleep(86400)
            START_FLAG=False
            today = datetime.datetime.utcnow().date() 
            yesterday = today - datetime.timedelta(days=1)
            two_days = today - datetime.timedelta(days=2)
            tomorrow = datetime.datetime.combine(today,datetime.datetime.min.time()) + datetime.timedelta(days=1)
            last_label_day = two_days

            last_true_label_timestamp = (last_label_day - datetime.date(1970, 1, 1)).total_seconds()
            yesterday_timestamp = (yesterday - datetime.date(1970, 1, 1)).total_seconds()
            today_timestamp = (today - datetime.date(1970, 1, 1)).total_seconds()

        elif sys.argv[1] == 'prod-cont':
            print("Prod cont mode")
            START_FLAG=False
            today = datetime.datetime.utcnow().date() 
            yesterday = today - datetime.timedelta(days=1)
            two_days = today - datetime.timedelta(days=2)
            tomorrow = datetime.datetime.combine(today,datetime.datetime.min.time()) + datetime.timedelta(days=1)

            #last_day_file = open("last_label_day.json","r")
            #last_label_day = json.loads(last_day_file.read())
            #last_label_day = datetime.datetime.strptime(last_label_day, "%Y-%m-%d").date()

            #last_true_label_timestamp = (last_label_day - datetime.date(1970, 1, 1)).total_seconds()

            check_query = "SELECT MAX(start_time) FROM PREDICTIONS WHERE true_label IS NOT NULL"
            last_true_label = app_connection.execute(check_query).first()
            if last_true_label is not None:
                last_true_label_timestamp = last_true_label[0]
                last_label_day = datetime.utcfromtimestamp(int(last_true_label_timestamp)).date()
            else:
                print("No predictions in database with true labels to be filled in. \
                        Either check predict stream or it is not yet time to make true labels again")
                

            yesterday_timestamp = (yesterday - datetime.date(1970, 1, 1)).total_seconds()
            today_timestamp = (today - datetime.date(1970, 1, 1)).total_seconds()

        elif sys.argv[1] == 'test':
            app_connection.close()
            app_engine.close()
            print("In test mode, not running labeler, sleeping 10 minutes")
            time.sleep(600)
            continue
    else:
    '''
    #try:
    #   last_day_file = open("last_label_day.json","r")
    #   last_label_day = json.loads(last_day_file.read())
    #   last_label_day = datetime.datetime.strptime(last_label_day, "%Y-%m-%d").date()

    check_query = "SELECT MAX(start_time) FROM PREDICTIONS WHERE true_label IS NOT NULL"
    last_true_label = app_connection.execute(check_query).first()[0]

    check_query2 = "SELECT MIN(start_time) FROM PREDICTIONS WHERE true_label IS NULL"
    first_empty_label = app_connection.execute(check_query2).first()[0]

    #if last_true_label is not None:
    #    last_true_label_timestamp = last_true_label
    #    last_label_day = datetime.datetime.utcfromtimestamp(int(last_true_label_timestamp)).date()
    if first_empty_label is not None:
    
        last_true_label_timestamp = first_empty_label
        last_label_day = datetime.datetime.utcfromtimestamp(int(last_true_label_timestamp)).date()
        print("First NULL true labels entry is at " + last_label_day.strftime("%m/%d/%Y"))

    else:
        print("No predictions in database with true labels to be filled in. \
                Check predict stream")

        print("Sleeping for 30 minutes")
        time.sleep(1800)
        continue

    today = datetime.datetime.utcnow().date() 
    yesterday = today - datetime.timedelta(days=1)
    two_days = today - datetime.timedelta(days=2)
    tomorrow = datetime.datetime.combine(today,datetime.datetime.min.time()) + datetime.timedelta(days=1)
    #except FileNotFoundError:
    #    print("Yesterday record file not found, troubleshooting required, assuming labels can be created now for yesterday")
    #    today = datetime.datetime.utcnow().date() 
    #    yesterday = today - datetime.timedelta(days=1)
    #    two_days = today - datetime.timedelta(days=2)
    #    tomorrow = datetime.datetime.combine(today,datetime.datetime.min.time()) + datetime.timedelta(days=1)
        #last_label_day = two_days
        # Enter code to query predictions database for last label day time instead, set day from there, save new file

    #    check_query = "SELECT MAX(end_time) FROM PREDICTIONS WHERE true_label IS NOT NULL"
    #    last_true_label = app_connection.execute(check_query).first()
    #    if last_true_label is not None:
    #        last_true_label_timestamp = last_true_label[0]
    #        last_label_day = datetime.utcfromtimestamp(int(last_true_label_timestamp)).date()
    #    else:
    #        print("No predictions in database with true labels to be filled in. \
    #                Either check predict stream or it is not yet time to make true labels again")

        #check_predicts = pd.read_sql_query(check_query, app_connection)
    
    yesterday_timestamp = (yesterday - datetime.date(1970, 1, 1)).total_seconds()
    today_timestamp = (today - datetime.date(1970, 1, 1)).total_seconds()
    # End else statement

    #check_query = "SELECT * FROM PREDICTIONS WHERE true_label IS NULL AND "
    
    #check_predicts = pd.read_sql_query(check_query, app_connection)

    prediction_query = "SELECT start_time,end_time FROM PREDICTIONS WHERE true_label IS NULL AND start_time BETWEEN " + str(last_true_label_timestamp) + " AND " + str(today_timestamp)
    prediction_times_df = pd.read_sql(prediction_query, app_connection)

    if last_label_day == yesterday:
        print("Not enough time has passed to label next batch of predictions")
        sleep_time = int((tomorrow - datetime.datetime.utcnow()).total_seconds())
        print("Sleeping for " + str(sleep_time/3600) + " hours until next day labeling process")
        time.sleep(sleep_time)
        continue

    # Check if there are more predictions in the prediction database, if not, loop
    elif prediction_times_df.empty:
        print("No predictions to generate true labels for this day, please check status of models and predict-stream")
        print("Sleeping for 1 hour")
        time.sleep(3600)
        continue

    elif last_label_day<=two_days:
        #print("Creating true labels for " + yesterday.strftime("%m/%d/%Y, %H:%M:%S"))
        print("Creating true labels for " + last_label_day.strftime("%m/%d/%Y"))

        ais_query = 'SELECT * FROM ais WHERE "timeOfFix" BETWEEN ' + str(last_true_label_timestamp) + " AND " + str(today_timestamp) 

        # Get all ais positions for yesterday
        ais_df = pd.read_sql(ais_query, app_connection)

        # Load previous day mmsi list from json
        # If file doesn't exist, then set equal to none
        try:
            f = open('/home/tracking/next_day.json')
            previous_day_dict = json.load(f)
            f.close()
        except FileNotFoundError:
            print("No next day carryover file found, assumning no carryover vessels")
            previous_day_dict = {None}
        if previous_day_dict=={}:
            previous_day_dict = {None}


        # Add lat long , brg, range data
        all_range_mmsi_times_df = pd.DataFrame()
        total_next_day_df = pd.DataFrame()
        for radius in radius_list:
            print("Processing true labels for " + str(radius))
            mmsi_times_df, next_day_df = get_times(ais_df, previous_day_dict, radius, today_timestamp)
            # Concatenate to total dataframes
            all_range_mmsi_times_df = pd.concat([all_range_mmsi_times_df, mmsi_times_df], ignore_index=True)
            total_next_day_df = pd.concat([total_next_day_df, next_day_df], ignore_index=True)

        
        # Save the next day mmsi list to a json (in case the program crashes, don't want to save in a dataframe)
        total_next_day_df.to_json('/home/tracking/next_day.json', orient='records')

        
        # Query predictions database for end time of last set of predictions that had
        #prediction_query = "SELECT start_time,end_time FROM PREDICTIONS WHERE start_time BETWEEN " + str(yesterday_timestamp) + " AND " + str(today_timestamp)
        #prediction_times_df = pd.read_sql(prediction_query, app_connection)
         
        # Iterate through times, create json of true labels for various radius, assign true labels for each time
        prediction_times_df['true_label'] = prediction_times_df.apply(lambda x: combine_classes(x['start_time'],x['end_time'],all_range_mmsi_times_df, radius_list), axis=1)
        
        # Iterate through dataframe and save true labels to each section
        prediction_times_df.apply(lambda x: insert_true_labels(x['true_label'],x['start_time'],x['end_time'], app_connection), axis=1)

        # save mmsi start/end times to database
        all_range_mmsi_times_df['record_timestamp'] = datetime.datetime.utcnow()
        all_range_mmsi_times_df.to_sql(name='ais_times', con=app_connection, index=False, if_exists='append')
        
        print("True labels applied and saved for " + yesterday.strftime("%m/%d/%Y, %H:%M:%S"))
        # Save date true labels made for to last_label_day.json
        #json_datetime = json.dumps(yesterday.isoformat())
        #jsonFile = open("last_label_day.json", "w")
        #jsonFile.write(json_datetime)
        #jsonFile.close()


        #app_connection.close()
        #app_engine.dispose()
