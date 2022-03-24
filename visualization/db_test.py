import sqlalchemy as db

print("Creating engine")
#engine = db.create_engine('mysql+mysqlconnector://mbari_user:mbariNPS1234!@#$@172.20.210.216:3306/mbari')
#engine = db.create_engine('mysql+mysqlconnector://mbari_admin:mbariNPS1234!@#$@pearl-new.uc.nps.edu:3306/mbari')

engine = db.create_engine('mysql+mysqlconnector://mbari_user:mbariNPS1234!@#$@pearl-new.uc.nps.edu:3306/mbari')

print("Creating connection")
connection = engine.connect()

query = "SELECT * FROM tmp"
print("Issuing Query")
result = connection.execute(query)

print(result.all())