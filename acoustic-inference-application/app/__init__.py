from flask import Flask
from config import *
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
import os
from flask_bootstrap import Bootstrap

app_settings = os.getenv('APP_SETTINGS') #or DevelopmentConfig
#app_settings = DevelopmentConfig
app = Flask(__name__)
app.config.from_object(app_settings)
db = SQLAlchemy(app)
migrate = Migrate(app, db, compare_type=True)
login = LoginManager(app)
login.login_view = 'login'
bootstrap = Bootstrap(app)

'''
# Attempt using bokeh server that can run with multiple ports
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.server.util import bind_sockets
from bokeh.server.server import BaseServer
from bokeh.server.tornado import BokehTornado
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from threading import Thread
#from visualization.ais_plot import ais_plot
#from visualization.spectrogram_scroller_plot import spectrogram_scroller_plot
from visualization.spectrogram_plot import spectrogram_plot
import asyncio
import os

#ais_plot = Application(FunctionHandler(ais_plot))
spectrogram_plot = Application(FunctionHandler(spectrogram_plot))
#spectrogram_scroller_plot = Application(FunctionHandler(spectrogram_scroller_plot))

# This is so that if this app is run using something like "gunicorn -w 4" then
# each process will listen on its own port


def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0
    
from app import settings
settings.init()
for p in range(5006, 5010):
    if is_port_in_use(p):
        continue
    else:
        sockets, settings.port = bind_sockets("localhost", p)
        break

def bk_worker():
    asyncio.set_event_loop(asyncio.new_event_loop())

    bokeh_tornado = BokehTornado({'/spectrogram_plot':spectrogram_plot}, \
        extra_websocket_origins=["127.0.0.1:5000", "127.0.0.1:"+str(settings.port), \
        "0.0.0.0:"+str(settings.port), "localhost:"+str(settings.port), "0.0.0.0:5000", \
            "localhost:5000", "172.21.0.1:5000", "172.21.0.1:"+str(settings.port), "172.21.0.2:5000", "172.21.0.2:"+str(settings.port), "*"])
    bokeh_http = HTTPServer(bokeh_tornado)
    bokeh_http.add_sockets(sockets)

    server = BaseServer(IOLoop.current(), bokeh_tornado, bokeh_http)
    server.start()
    server.io_loop.start()

t = Thread(target=bk_worker)
t.daemon = True
t.start()
'''
from app import routes, models
