from app import app, db
from app.models import Predictions, User, Ais, Mmsi, Models, Ship_Classes, Ais_Times

@app.shell_context_processor
def make_shell_context():
    return {'db': db, 'User': User, 'AIS': Ais, 'Prediction': Predictions, 'MMSI':Mmsi, 'Models': Models, 'Ship_Classes':Ship_Classes, 'Ais_Times':Ais_Times}