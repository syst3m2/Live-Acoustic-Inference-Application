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

engine = db.create_engine(SQLALCHEMY_DATABASE_URI)
connection = engine.connect()

latitude = 36.712465
longitude = -122.187548

query = "SELECT * FROM ais"

all_ships = pd.read_sql_query(query, connection)

print("range longitude")
print(all_ships['longitude'].min())
print(all_ships['longitude'].max())

print("original bearing")
print(all_ships['bearing_from_sensor'].head())
print("original distance")
print(all_ships['dist_from_sensor_km'].head())

print("Updating range")

print("Average distance before update")
print(all_ships['dist_from_sensor_km'].mean())
all_ships['dist_from_sensor_km'] = all_ships.apply(lambda x: compare_lat_long(latitude, longitude, x.latitude, x.longitude), axis=1)
print("Average distance after update")
print(all_ships['dist_from_sensor_km'].mean())

print("Average bearing before update")
print(all_ships['bearing_from_sensor'].mean())
all_ships['bearing_from_sensor'] = all_ships.apply(lambda x: get_bearing(float(latitude), float(longitude), x['latitude'], x['longitude']), axis=1)
print("Average bearing after update")
print(all_ships['bearing_from_sensor'].mean())

all_ships.to_sql('tmp_table', engine, if_exists='replace', index=False)

verify_tmp = "SELECT * FROM tmp_table"

verify_table = pd.read_sql_query(verify_tmp, connection)

print("Data from tmp table")
print(verify_table['dist_from_sensor_km'].head())
print(verify_table['bearing_from_sensor'].head())

print("updating AIS table")

sql = """
    UPDATE ais AS f
    SET bearing_from_sensor = t.bearing_from_sensor,
    dist_from_sensor_km = t.dist_from_sensor_km
    FROM tmp_table AS t
    WHERE f.id = t.id
"""

connection.execute(sql)


all_ships_check = pd.read_sql_query(query, connection)

print("Check information is the same now in AIS table")
print("Old average and max distance")
print(all_ships['dist_from_sensor_km'].mean())
print(all_ships['dist_from_sensor_km'].max())

print("New average and max distance")
print(verify_table['dist_from_sensor_km'].mean())
print(verify_table['dist_from_sensor_km'].max())

print("verify average and max distance")
print(all_ships_check['dist_from_sensor_km'].mean())
print(all_ships_check['dist_from_sensor_km'].max())

print("-------------------------------------")

print("Old Average and max bearing")
print(all_ships['bearing_from_sensor'].mean())
print(all_ships['bearing_from_sensor'].max())

print("New Average and max bearing")
print(verify_table['bearing_from_sensor'].mean())
print(verify_table['bearing_from_sensor'].max())

print("Verify Average and max bearing")
print(all_ships_check['bearing_from_sensor'].mean())
print(all_ships_check['bearing_from_sensor'].max())


print("-------------------------------------")

print("Old distances and bearings")
print(all_ships['bearing_from_sensor'].head())
print(all_ships['dist_from_sensor_km'].head())

print("New distances and bearings")
print(verify_table['bearing_from_sensor'].head())
print(verify_table['dist_from_sensor_km'].head())

print("verify distances and bearings")
print(all_ships_check['bearing_from_sensor'].head())
print(all_ships_check['dist_from_sensor_km'].head())



# Remove table

drop = "DROP TABLE tmp_table"

connection.execute(drop)

connection.close()
engine.dispose()
