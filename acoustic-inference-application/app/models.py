from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from app import db, login
from sqlalchemy.dialects.postgresql import JSON

@login.user_loader
def load_user(id):
    return User.query.get(int(id))

# To update database 


class User(UserMixin, db.Model):
    __tablename__ = 'app_users'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    models = db.relationship('Models', backref='author', lazy='dynamic')

    def __repr__(self):
        return '<user {}>'.format(self.username)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
# 'mmsi', 'imoNumber', 'name', 'callSign', 'cargo', 'heading', 'navStatus', 'SOG', 
# 'latitude', 'longitude', 'timeOfFix', 'dist_from_sensor_km', 'dead_weight', 'length', 'beam', 'desig', 
# 'merc_latitude', 'merc_longitude','ship_class','bearing_from_sensor'
# COG = db.Column(db.Integer)
class Ais(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    mmsi = db.Column(db.Integer, db.ForeignKey('mmsi.mmsi'), index=True)
    imoNumber = db.Column(db.Integer, index=True)
    name = db.Column(db.String(140), index=True)
    callSign = db.Column(db.String(140), index=True)
    cargo = db.Column(db.String(140))
    heading = db.Column(db.Float)
    navStatus = db.Column(db.String(140))
    SOG = db.Column(db.Float)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    timeOfFix = db.Column(db.Integer, index=True)
    dist_from_sensor_km = db.Column(db.Float)
    dead_weight = db.Column(db.Float)
    length = db.Column(db.Integer)
    beam = db.Column(db.Integer)
    desig = db.Column(db.String(140), index=True)
    merc_latitude = db.Column(db.Float)
    merc_longitude = db.Column(db.Float)
    ship_class = db.Column(db.String(140), index=True) #Labels the class of ship, models use this to compare predictions
    bearing_from_sensor = db.Column(db.Float)
    record_timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)

# Maybe change integer time columns to datetimes or BIGINT
# Records the .wav filepath location, time data, true labels as derived from AIS, and predicted labels from any models 
class Predictions(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    start_time = db.Column(db.Integer, index=True)
    end_time = db.Column(db.Integer, index=True)
    #true_label = db.Column(db.String(512), index=True) # Variable length field containing true labels as determined from AIS stream
    true_label = db.Column(db.Text)
    #true_label = db.Column(db.JSON, index=True)
    #model_predictions = db.Column(db.String(512), index=True) # Need to change to JSON
    #model_predictions = db.Column(JSON, index=True)
    model_predictions = db.Column(db.Text)
    # change to both true labels and predictions to db.String(4294000000) for longtext data type
    # or look into json data types db.JSON() but only works for postgres not sqlite
    record_timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)

class Models(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    file_name = db.Column(db.String(140), index=True)
    model_name = db.Column(db.String(140), index=True)
    model_type = db.Column(db.String(140), index=True)
    channels = db.Column(db.Integer, index=True)
    model_input = db.Column(db.String(140), index=True)
    model_choice = db.Column(db.String(140), index=True) #resnet1, cnn_model, dev_bnn_model, resnet_bnn, simple_bnn_vi_model, cnn_model_hparam, dev_model_hparam 
    prediction_classes = db.Column(db.String(256), index=True)
    params = db.Column(db.String(16384)) #Need to change to JSON
    training_accuracy_score = db.Column(db.Float, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('app_users.id'))
    active = db.Column(db.Boolean, index=True) #status either running or not
    record_timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)


    def __repr__(self):
        return '<Models {}>'.format(self.id, self.model_name, self.model_type)

class Mmsi(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    mmsi = db.Column(db.Integer, index=True, unique=True, primary_key=True)
    dead_weight = db.Column(db.Float)
    length = db.Column(db.Integer)
    beam = db.Column(db.Integer)
    desig = db.Column(db.String(140), index=True)
    #ship_class = db.Column(db.String(140), index=True) # Creates label for machine learning predictions, Class A, B, C, D, E
    record_timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)

    # Create additional parameter for ais track table
    ais = db.relationship('Ais', backref='mmsi_metadata', lazy='dynamic')
    #ais_times = db.relationship('Ais_Times', backref='mmsi_metadata', lazy='dynamic')


class Ship_Classes(db.Model):
    __tablename__ = 'ship_classes'
    # Reference table to translate ship types to their labels to predict
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    desig = db.Column(db.String(512), index=True, unique=True)
    ship_class = db.Column(db.String(512), index=True)
    

class Ais_Times(db.Model):
    __tablename__ = 'ais_times'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    mmsi = db.Column(db.Integer, db.ForeignKey('mmsi.mmsi'), index=True)
    radius = db.Column(db.Integer)
    start_time = db.Column(db.Integer, index=True)
    end_time = db.Column(db.Integer, index=True)
    cpa = db.Column(db.Float)
    cpa_time = db.Column(db.Integer)
    desig = db.Column(db.String(140), index=True)
    ship_class = db.Column(db.Text, index=True)
    record_timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
