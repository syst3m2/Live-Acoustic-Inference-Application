import time
import datetime
import pandas as pd
import sqlalchemy as db
from os import path
from ais_streamer import *
import os
from urllib3.exceptions import NewConnectionError
from requests.exceptions import ConnectionError
import sys
import time

# Get environment variables
# make sure to update API key
SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') #or 'sqlite:///../acoustic-inference-application/app.db'

SEAVISION_API_KEY = os.environ.get('SEAVISION_API_KEY') #or 'VLpP8iMKys7MvxmAqKsRf7WV2HZ71byO51HLmIjm'

FIRST_FLAG = True
MERGE_FAIL = False

# Initialize database connection

engine = db.create_engine(SQLALCHEMY_DATABASE_URI)
connection = engine.connect()

query_latitude = 36.712465
query_longitude = -122.187548

while True:
    # If the file exists, grab the last query of ships to compare to current query
    # else get the last time from the database and calculate time differences, resave json
    if MERGE_FAIL:
        print("Merge failed, setting hours to 1 to continue AIS stream")
        now = datetime.datetime.utcnow()
        minus_hour = now - datetime.timedelta(hours=1)
        latest_time_timestamp = (minus_hour - datetime.datetime(1970, 1, 1)).total_seconds()
        hours=1
        last_query_df = pd.DataFrame()
        MERGE_FAIL = False

    elif path.exists("/home/tracking/last_query.json"):
        last_query_df = pd.read_json('/home/tracking/last_query.json')
        last_query_df['ship_class'] = last_query_df['ship_class'].astype('object')
        latest_time_timestamp = last_query_df['timeOfFix'].max()
        latest_time = datetime.datetime.utcfromtimestamp(latest_time_timestamp)
        # Calculate time differences from now to latest time in last query to use as times for next query
        now = datetime.datetime.utcnow()

        time_delta = now - latest_time

        hours = (time_delta.seconds//3600 + 1)  + (time_delta.days*24)

        # Use for queries that requires days as input (track history)
        days = time_delta.days + 1

        if days > 90:
            days = 90
    else:
        print("No last_query json file exists. Assuming start now")
        now = datetime.datetime.utcnow()
        minus_hour = now - datetime.timedelta(hours=1)
        latest_time_timestamp = (minus_hour - datetime.datetime(1970, 1, 1)).total_seconds()
        hours=1
        last_query_df = pd.DataFrame()
    

    if hours > 2160:
        hours = 2160
    elif hours<=0:
        hours = 1
    
    if FIRST_FLAG:
        if sys.argv[1]=='dev-first':
            hours = 1
            FIRST_FLAG=False

        elif sys.argv[1]=='dev-cont':
            FIRST_FLAG=False

        if sys.argv[1]=='prod-first':
            hours = 1
            FIRST_FLAG=False

        elif sys.argv[1]=='prod-cont':
            FIRST_FLAG=False
            
        elif sys.argv[1]=='test':
            print("In test mode, not running labeler, sleeping 10 minutes")
            time.sleep(600)
            continue

    # Query for ship positions and tracks within specified timeframes
    # Issues too many queries and overloads data, so not viable solution for stream
    start = time.perf_counter()
    print("querying AIS data for past " + str(hours) + " hours...")
    if hours > 1: 
        try:
           discard_ships_df, ships_df = ais_boundary_track_query(latitude = query_latitude, longitude = query_longitude, days=days, hours = hours, radius = 300, latest_time=latest_time_timestamp, engine=engine, connection=connection)
        except ConnectionError:
           print("No internet connection, trying again...")
           continue
        except RuntimeError as e:
            print("Runtime Error, Rate Limit likely exceeded. Waiting for 1 hour and retrying")
            print(e)
            sleep(900)
            continue
        except Exception as e:
            print("Exception during AIS Boundary Track API request, trying again...")
            print(e)
            continue
    
    else:
        try:
            ships_df = ais_boundary_query(latitude = query_latitude, longitude = query_longitude, hours = hours, radius = 100, engine=engine, connection=connection)
            ships_df = ships_df.drop('COG', axis=1)
        except ConnectionError:
            print("No internet connection, trying again")
            continue
        except RuntimeError:
            print("Rate limit exceeded, waiting 15 minutes and trying again...")
            time.sleep(900)
            continue
        except Exception as e:
            print("Exception during AIS Boundary Query API request, trying again...")
            print(e)
            continue
    stop = time.perf_counter()
    print("query complete!")
    print("Total time to make query: " + str((stop-start)/60) + " minutes")

    ships_df = ships_df[ships_df['timeOfFix'] > latest_time_timestamp]

    # Compare newest dataframe to dataframe where last_query=True, only keep newer positions not already in database
    if not last_query_df.empty:
        last_query_df = last_query_df.drop('record_timestamp', axis=1)
        last_query_df['SOG'] = last_query_df['SOG'].astype(float)
        last_query_df['heading'] = last_query_df['heading'].astype(float)
        last_query_df['mmsi'] = last_query_df['mmsi'].astype(int)
        last_query_df['dead_weight'] = last_query_df['dead_weight'].astype(int)
        last_query_df['length'] = last_query_df['length'].astype(int)
        last_query_df['beam'] = last_query_df['beam'].astype(int)
        last_query_df['imoNumber'] = last_query_df['imoNumber'].astype(int)
        try:
            #print(ships_df.dtypes)
            #print(last_query_df.dtypes)
            new_ship_pos = ais_compare(old_ais_df=last_query_df, new_ais_df=ships_df)
        except ValueError:
            print("Merge failed on the following dataframes")
            print("Last query dataframe:")
            print(last_query_df)
            print("Current query dataframe:")
            print(ships_df)
            print("Saving new ship query to json for inspection")
            ships_df.to_json('merge_fail.json')
            print("Skipping database store and conducting query again")
            MERGE_FAIL = True
            continue
    else:
        new_ship_pos = ships_df

    # If ship enters within 30km or 20km (need to ask which to do), mark "enter predict zone"
    # If ship exits predict zone, then mark exited
    
    #should label with in_pred_zone/not_in_pred_zone
    # mark enter logic
    # if position is < prediction distance
    # query for previous entries of same mmsi marked with exit and enter, if latest time is an exit, 
    # then mark enter again. else if last was marked enter, then don't label. 
    # if no previous mmsi entry, then mark enter

    # mark exit logic
    # query mmsi to see if previously marked enter, if last marked enter and now 
    # distance is > prediction distance, then mark exit

    # Enter new positions in database with last_query=True
    if not new_ship_pos.empty:
        ais_db_write(new_ship_pos, engine)
        new_ship_pos.to_json('/home/tracking/last_query.json')
        print('New positions saved')
    else:
        print('No new positions')
    
    time.sleep(45)
