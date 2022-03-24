import datetime
from pandas.io.parsers import CParserWrapper
import sqlalchemy as db
import pandas as pd
import os
import json
import sys
import time

def insert_id(id,start_time,end_time, radius, cpa_time, connection):
    try:
        insert_statement = "UPDATE ais_times SET id=" + str(id) + " WHERE start_time=" + str(start_time) + " AND end_time=" + str(end_time) + " AND radius=" + str(radius) + " AND cpa_time=" + str(cpa_time)
    except TypeError:
        print(true_labels)
        print(start_time)
        print(end_time)
        input("Check type error data")
    app_connection.execute(insert_statement)

    return

app_db = os.environ.get('DATABASE_URL')
app_engine = db.create_engine(app_db)
app_connection = app_engine.connect()

'''
query = "SELECT * FROM ais_times"

ais_times_df = pd.read_sql_query(query, app_connection)
ais_times_df.to_csv('ais_times.csv', index=False)

#ais_times_df['id'] = ais_times_df.index + 1

#ais_times_df.apply(lambda x: insert_id(x['id'],x['start_time'],x['end_time'],x['radius'],x['cpa_time'], app_connection), axis=1)
'''

ais_times_df = pd.read_csv('ais_times.csv')

print(ais_times_df.head())

ais_times_df.to_sql(name='ais_times', con=app_connection, index=False, if_exists='append')

print("Finished")
app_connection.close()
app_engine.dispose()