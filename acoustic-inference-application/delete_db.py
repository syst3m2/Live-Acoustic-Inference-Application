import sqlalchemy as db
import os
import pandas as pd

POSTGRESDB = 'postgresql://postgres:postgres@app-db:5432/inference_db'

pg_engine = db.create_engine(POSTGRESDB)
pg_connection = pg_engine.connect()
print("Connected to postgresql")

print("Deleting data from Postgresql")
'''
delete_ais = "DELETE FROM AIS"
delete_predict = "DELETE FROM PREDICTIONS"
delete_models = "DELETE FROM MODELS"
delete_users = "DELETE FROM USER"

pg_connection.execute(delete_ais)
pg_connection.execute(delete_predict)
pg_connection.execute(delete_models)
pg_connection.execute(delete_users)
'''

#drop = "DROP TABLE ais, predictions, models, user"
drop = "DROP SCHEMA IF EXISTS public CASCADE"
pg_connection.execute(drop)

create = "CREATE SCHEMA public"
pg_connection.execute(create)

grant = "GRANT ALL ON SCHEMA public TO postgres"
pg_connection.execute(grant)
grant2 = "GRANT ALL ON SCHEMA public TO public"
pg_connection.execute(grant2)

pg_connection.close()
pg_engine.dispose()
