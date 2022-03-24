import sqlalchemy as db
import os
import pandas as pd
import numpy as np
import datetime

POSTGRESDB = 'postgresql://postgres:postgres@app-db:5432/inference_db'
#SQLITEDB = 'sqlite:////home/acoustic_app/app.db' #os.environ.get('DATABASE_URL')

pg_engine = db.create_engine(POSTGRESDB)
pg_connection = pg_engine.connect()
print("Connected to postgresql")

#lite_engine = db.create_engine(SQLITEDB)
#lite_connection = lite_engine.connect()
#print("Connected to sqlite")

#ais_query = 'SELECT * FROM AIS'
#predict_query = 'SELECT * FROM PREDICTIONS'
#model_query = 'SELECT * FROM MODELS'
#user_query = 'SELECT * FROM USER'
#mmsi_query = 'SELECT * FROM MMSI'

#ais_data = pd.read_sql_query(ais_query, lite_engine)
#predict_data = pd.read_sql_query(predict_query, lite_engine)
#model_data = pd.read_sql_query(model_query, lite_engine)
#user_data = pd.read_sql_query(user_query, lite_engine)

#model_data['record_timestamp'] = pd.to_datetime(model_data['record_timestamp'],infer_datetime_format=True)

#model_data['record_timestamp'] = model_data.record_timestamp.values.astype(np.int64) // 10 ** 9


#predict_data['record_timestamp'] = pd.to_datetime(predict_data['record_timestamp'],infer_datetime_format=True)

#predict_data['record_timestamp'] = predict_data.record_timestamp.values.astype(np.int64) // 10 ** 9

#mmsi_data = ais_data.groupby('mmsi').first().reset_index()

#mmsi_data = mmsi_data[['mmsi', 'dead_weight', 'length', 'beam', 'desig', 'ship_class', 'record_timestamp']]
#mmsi_data['id'] = mmsi_data.index + 1

#print("From sqlite")
#print(ais_data.head())
#print(predict_data.head())
#print(model_data.head())
#print(user_data.head())
#print(mmsi_data.head())

'''
print("Deleting data from Postgresql")
delete_ais = "DELETE FROM AIS"
delete_predict = "DELETE FROM PREDICTIONS"
delete_models = "DELETE FROM MODELS"
delete_users = "DELETE FROM USER"

pg_connection.execute(delete_ais)
pg_connection.execute(delete_predict)
pg_connection.execute(delete_models)
pg_connection.execute(delete_users)
'''
# Initialize database with ship class dictionary
ship_class_dict ={'Landings Craft':'Class A', 'Military ops':'Class A','Fishing vessel':'Class A','Fishing Vessel':'Class A' ,'Fishing Support Vessel':'Class A', 'Tug':'Class A', 'Pusher Tug':'Class A', 'Dredging or UW ops':'Class A', 'Towing vessel':'Class A', 'Crew Boat':'Class A', 'Buoy/Lighthouse Vessel':'Class A', 'Salvage Ship':'Class A', 'Research Vessel':'Class A', 'Anti-polution':'Class A', 'Offshore Tug/Supply Ship':'Class A', 'Law enforcment':'Class A', 'Landing Craft':'Class A', 'SAR':'Class A', 'Patrol Vessel':'Class A', 'Pollution Control Vessel': 'Class A', 'Offshore Support Vessel':'Class A',
                        'Pleasure craft':'Class B', 'Yacht':'Class B', 'Sailing vessel':'Class B', 'Pilot':'Class B', 'Diving ops':'Class B', 
                        'Passenger (Cruise) Ship':'Class C', 'Passenger Ship':'Class C', 'Passenger ship':'Class C', 'Training Ship': 'Class C',
                        'Naval/Naval Auxiliary':'Class D','DDG':'Class D','LCS':'Class D','Hospital Vessel':'Class D' ,'Self Discharging Bulk Carrier':'Class D' ,'Cutter':'Class D', 'Passenger/Ro-Ro Cargo Ship':'Class D', 'Heavy Load Carrier':'Class D', 'Vessel (function unknown)':'Class D',
                        'General Cargo Ship':'Class D','Wood Chips Carrier':'Class D', 'Bulk Carrier':'Class D' ,'Cement Carrier':'Class D','Vehicles Carrier':'Class D','Cargo ship':'Class D', 'Oil Products Tanker':'Class D', 'Ro-Ro Cargo Ship':'Class D', 'USNS RAINIER':'Class D', 'Supply Tender':'Class D', 'Cargo ship':'Class D', 'LPG Tanker':'Class D', 'Crude Oil Tanker':'Class D', 'Container Ship':'Class D', 'Container ship':'Class D','Bulk Carrier':'Class D', 'Chemical/Oil Products Tanker':'Class D', 'Refrigerated Cargo Ship':'Class D', 'Tanker':'Class D', 'Car Carrier':'Class D', 'Deck Cargo Ship' :'Class D', 'Livestock Carrier': 'Class D',
                        'Bunkering Tanker':'Class D', 'Water Tanker': 'Class D', 'FSO': 'Class D', 
                        'not ship':'Class E' }

desig_list=[]
class_list=[]

for key in ship_class_dict:
    desig_list.append(key)
    class_list.append(ship_class_dict[key])

new_ship_class_dict = {'desig':desig_list, 'ship_class':class_list}

ship_dict = pd.DataFrame(data=new_ship_class_dict)

#mmsi_data = mmsi_data.drop('ship_class', axis=1)

#model_data['active'] = True

print("Storing new data in PostgreSQL")
#user_data.to_sql(name='app_users', con=pg_engine, index=False, if_exists='append')
#mmsi_data.to_sql(name='mmsi', con=pg_engine, index=False, if_exists='append')
#ais_data.to_sql(name='ais', con=pg_engine, index=False, if_exists='append')
#predict_data.to_sql(name='predictions', con=pg_engine, index=False, if_exists='append')
#model_data.to_sql(name='models', con=pg_engine, index=False, if_exists='append')
ship_dict.to_sql(name='ship_classes', con=pg_engine, index=False, if_exists='append')

# Initializing sequence value in database
#user_seq = "SELECT setval('public.app_users_id_seq', (SELECT MAX(id) FROM app_users))"
#mmsi_seq = "SELECT setval('public.mmsi_id_seq', (SELECT MAX(id) FROM mmsi))"
#ais_seq = "SELECT setval('public.ais_id_seq', (SELECT MAX(id) FROM ais))"
#predict_seq = "SELECT setval('public.predictions_id_seq', (SELECT MAX(id) FROM predictions))"
#model_seq = "SELECT setval('public.models_id_seq', (SELECT MAX(id) FROM models))"
ship_class_seq = "SELECT setval('public.ship_classes_id_seq', (SELECT MAX(id) FROM ship_classes))"

#pg_connection.execute(user_seq)
#pg_connection.execute(mmsi_seq)
#pg_connection.execute(ais_seq)
#pg_connection.execute(predict_seq)
#pg_connection.execute(model_seq)


print("Verifying data in PostgreSQL")
#ais_data = pd.read_sql_query(ais_query, pg_engine)
#predict_data = pd.read_sql_query(predict_query, pg_engine)
#model_data = pd.read_sql_query(model_query, pg_engine)
#user_data = pd.read_sql_query(user_query, pg_engine)
#mmsi_data = pd.read_sql_query(mmsi_query, pg_engine)

#print("From postgresql")
#print(ais_data.head())
#print(predict_data.head())
#print(model_data.head())
#print(user_data.head())
#print(mmsi_data.head())


pg_connection.close()
pg_engine.dispose()

#lite_connection.close()
#lite_engine.dispose()