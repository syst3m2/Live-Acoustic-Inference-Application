import os
basedir = os.path.abspath(os.path.dirname(__file__))

class BaseConfig:
    TESTING = False

class Config(BaseConfig):
    SECRET_KEY = os.environ.get('SECRET_KEY') #or 'you-will-never-guess'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') #or 'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    #ACOUSTIC_DATABASE_URI = os.environ.get('ACOUSTIC_DATABASE_URL') #or '/home/lemgog/thesis/acoustic_app/data/mbari/master_index.db'
    MODEL_FOLDER = os.environ.get('MODEL_FOLDER') #or '/home/models' 
    
class ProductionConfig(BaseConfig):
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')