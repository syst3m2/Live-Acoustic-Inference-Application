from flask import render_template, flash, redirect, request, url_for, jsonify, Response, send_file, send_from_directory
from app import app, db, login
from app.forms import LoginForm, RegistrationForm, ModelUploadForm, ModelDeleteForm, ModelActivateForm, ShipClassUpdateForm, ChangeShipClassForm
from flask_login import current_user, login_user, logout_user, login_required
from app.models import User, Models, Ais, Mmsi, Ship_Classes
from werkzeug.urls import url_parse
import pandas as pd
import sys
sys.path.append("..")
import requests
import re
from app import settings
from app.models import Predictions
import json
from bokeh.embed import server_session, server_document
from bokeh.client import pull_session
import datetime
from flask_login import LoginManager
from flask import g
import os

@login.user_loader
def load_user(user_id):
    return User.query.get(user_id)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                          'favicon.ico',mimetype='image/vnd.microsoft.icon')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', title='Villemez Thesis - Live Acoustic Inference')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(next_page)
    return render_template('login.html', title='Sign In', form=form)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route('/models', methods=['GET', 'POST'])
def models():
    
    acoustic_plot = server_document(os.environ.get('INFERENCE_VIZ'))

    acoustic_plot = acoustic_plot.replace("visualization", "localhost")
    #acoustic_plot = acoustic_plot.replace("visualization", "GSOIS-B13-13")

    return render_template('models.html', title='Villemez Thesis - Live Acoustic Inference', acoustic_plot=acoustic_plot)


@app.route('/metrics', methods=['GET', 'POST'])
def metrics():

    accuracy_scores = server_document(os.environ.get('METRICS_VIZ'))

    accuracy_scores = accuracy_scores.replace("visualization", "localhost")

    return render_template('metrics.html', title='Model Performance Metrics', 
        accuracy_scores=accuracy_scores)


from werkzeug.utils import secure_filename
@app.route('/manage_models', methods=['GET', 'POST'])
@login_required
def manage_models():

    model_upload_form = ModelUploadForm()

    if "submit_upload" in request.form and model_upload_form.validate_on_submit():
        # Validate input data
        # Save filename and other data to database
        # grab model ID, create new folder, save files to folder

        model_filename = secure_filename(model_upload_form.modelfile.data.filename)
        '''
        if not model_filename.endswith('.h5'):
            flash('Only .h5 checkpoint files are compatible with this application')
            return redirect(url_for('manage_models'))
        '''

        if model_upload_form.training_accuracy_score.data > 1 or model_upload_form.training_accuracy_score.data < 0:
            flash('Invalid accuracy score, must be between 0 and 1')
            return redirect(url_for('manage_models'))

        params_filename = secure_filename(model_upload_form.paramsfile.data.filename)
        try:
            # Use this once string field changed to json
            params_data = json.loads(model_upload_form.paramsfile.data.read())
            params_data = json.dumps(params_data)
        except:
            flash("params.txt formatted incorrectly, please ensure JSON formatting")
            return redirect(url_for('manage_models'))
        
        
        model = Models(file_name=model_filename,model_name=model_upload_form.model_name.data, \
                        model_type=model_upload_form.model_type.data, channels=model_upload_form.channels.data, \
                        model_input=model_upload_form.model_input.data, model_choice=model_upload_form.model_choice.data, \
                        prediction_classes=model_upload_form.prediction_classes.data, params=params_data, \
                        training_accuracy_score=model_upload_form.training_accuracy_score.data, active=True, user_id=current_user.get_id(), record_timestamp=datetime.datetime.utcnow())

        db.session.add(model)
        db.session.commit()

        model_filepath = app.config['MODEL_FOLDER'] + '/id_' + str(model.id)
        model_filename = os.path.join(model_filepath, model_filename)
        params_filename = os.path.join(model_filepath, params_filename)
        os.makedirs(os.path.dirname(model_filename), exist_ok=True)
        os.makedirs(os.path.dirname(params_filename), exist_ok=True)

        model_upload_form.modelfile.data.save(model_filename)

        params_data
        f = open(params_filename, "w")
        f.write(params_data)
        f.close()
        
        flash('Model Deployed!')

        # Add code for if model save fails, delete from database

    model_delete_form = ModelDeleteForm()

    
    if "submit_delete" in request.form and model_delete_form.validate_on_submit():
        
        model_id = model_delete_form.delete_field.data.split(' ')[0] 
        model_query = "SELECT * FROM models WHERE user_id=" + str(current_user.get_id()) + " AND id='" + model_id + "'"
        model_to_delete = db.session.query(Models).filter(Models.user_id==str(current_user.get_id()),Models.id==model_id).first()
        #model_to_delete = Models.query.filter_by(Models.user_id==str(current_user.get_id()),Models.id==model_id).first()
        #model_to_delete = pd.read_sql(model_query, db.engine)
        if model_to_delete is None:
            flash("You do not own this model, please only select models you deployed")
            return redirect(url_for('manage_models'))
        else:
            model_to_delete.active = False
            db.session.add(model_to_delete)
            db.session.commit()

            flash(str(model_to_delete.model_name) + " stopped")

    model_activate_form = ModelActivateForm()

    if "submit_activate" in request.form and model_activate_form.validate_on_submit():
        
        model_id = model_activate_form.activate_field.data.split(' ')[0] 
        model_query = "SELECT * FROM models WHERE user_id=" + str(current_user.get_id()) + " AND id='" + model_id + "'"
        model_to_activate = db.session.query(Models).filter(Models.user_id==str(current_user.get_id()),Models.id==model_id).first()
        
        if model_to_activate is None:
            flash("You do not own this model, please only select models you deployed")
            return redirect(url_for('manage_models'))
        else:
            #print('Stopping ' + model_to_delete['model_name'][0])

            model_to_activate.active = True
            db.session.add(model_to_activate)
            db.session.commit()

            #flash(model_to_delete['model_name'][0] + " stopped")
            flash(str(model_to_activate.model_name) + " reactivated")

    # Add code to download models from the database

    return render_template('manage_models.html', title='Deploy or Delete Models', upload_form=model_upload_form, delete_form=model_delete_form, activate_form=model_activate_form)

from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.plotting import figure
from bokeh.layouts import widgetbox, column
from bokeh.embed import components
@app.route('/manage_data', methods=['GET', 'POST'])
@login_required
def manage_data():

    class_query = "SELECT * FROM ship_classes WHERE ship_class='Unknown'"
    desig_sizes = pd.read_sql(class_query, db.engine)

    avg_class_query = "SELECT ship_class,AVG(dead_weight),AVG(length),AVG(beam) FROM ais WHERE dead_weight != -1 AND dead_weight IS NOT NULL AND length != -1 AND length IS NOT NULL AND beam != -1 AND beam IS NOT NULL GROUP BY ship_class"
    class_sizes = pd.read_sql(avg_class_query, db.engine)
    class_sizes.columns = ['ship_class', 'avg_dead_weight', 'avg_length', 'avg_beam']

    avg_desig_query = "SELECT * FROM ship_classes WHERE ship_class!='Unknown'"
    known_desig_sizes = pd.read_sql(avg_desig_query, db.engine)

    # This function returns the average sizes of each ship designation that is categorized as Unknown
    def mmsi_data_query(desig, feature, engine):
        #if type == 'desig':
        query = "SELECT " + feature + " FROM mmsi WHERE desig='" + desig + "'"
        #elif type == 'ship_class':
        #    query = "SELECT " + feature + " FROM ais WHERE ship_class='" + desig + "'"

        size_data = pd.read_sql(query, engine)

        # Discard values for unknown size data that would skew information, calculate average sizes
        size_data = size_data[((size_data[feature]!=-1) & (size_data[feature].notna()))]
        #size_data = size_data[((size_data['dead_weight']!=-1) & (size_data['dead_weight'].notna()))]
        #size_data = size_data[((size_data['length']!=-1) & (size_data['length'].notna()))]
        #size_data = size_data[((size_data['beam']!=-1) & (size_data['beam'].notna()))]
        #size_data = size_data.groupby('desig').mean()
        size_data = size_data[feature].mean()

        return size_data

    if not desig_sizes.empty:
        desig_sizes['avg_dead_weight'] = desig_sizes.apply(lambda x: mmsi_data_query(x['desig'],'dead_weight', db.engine), axis=1)
        desig_sizes['avg_length'] = desig_sizes.apply(lambda x: mmsi_data_query(x['desig'],'length', db.engine), axis=1)
        desig_sizes['avg_beam'] = desig_sizes.apply(lambda x: mmsi_data_query(x['desig'],'beam', db.engine), axis=1)

    if not known_desig_sizes.empty:
        known_desig_sizes['avg_dead_weight'] = known_desig_sizes.apply(lambda x: mmsi_data_query(x['desig'],'dead_weight', db.engine), axis=1)
        known_desig_sizes['avg_length'] = known_desig_sizes.apply(lambda x: mmsi_data_query(x['desig'],'length', db.engine), axis=1)
        known_desig_sizes['avg_beam'] = known_desig_sizes.apply(lambda x: mmsi_data_query(x['desig'],'beam', db.engine), axis=1)

    # Average sizes of unknown designations
    data = ColumnDataSource(desig_sizes)

    columns = [
        TableColumn(field="desig", title="Ship Designation"),
        TableColumn(field="avg_dead_weight", title="Average Dead Weight"),
        TableColumn(field="avg_length", title="Average Length"),
        TableColumn(field="avg_beam", title="Average Beam"),
    ]

    data_table = DataTable(source=data, columns=columns)
    script, div = components(column(data_table))

    # Show the average sizes of the ship classes
    class_data = ColumnDataSource(class_sizes)

    class_columns = [
        TableColumn(field="ship_class", title="Ship Class"),
        TableColumn(field="avg_dead_weight", title="Average Dead Weight"),
        TableColumn(field="avg_length", title="Average Length"),
        TableColumn(field="avg_beam", title="Average Beam"),
    ]

    class_data_table = DataTable(source=class_data, columns=class_columns)
    class_script, class_div = components(column(class_data_table))

    # Show the average sizes of the known designations, in case change required
    known_desig_data = ColumnDataSource(known_desig_sizes)

    known_desig_columns = [
        TableColumn(field="desig", title="Ship Designation"),
        TableColumn(field="avg_dead_weight", title="Average Dead Weight"),
        TableColumn(field="avg_length", title="Average Length"),
        TableColumn(field="avg_beam", title="Average Beam"),
        TableColumn(field="ship_class", title="Ship Class"),
    ]

    known_desig_data_table = DataTable(source=known_desig_data, columns=known_desig_columns)
    known_desig_script, known_desig_div = components(column(known_desig_data_table))

    # Create and validate forms to update the class sizes and values
    class_form = ShipClassUpdateForm()

    if "submit_unknown_class" in request.form and class_form.validate_on_submit():
        ship_desig = class_form.desig_field.data
        new_class = class_form.class_select_field.data

        # Update ship class table
        class_update = db.session.query(Ship_Classes).filter(Ship_Classes.desig==ship_desig).first()
        class_update.ship_class = new_class

        db.session.add(class_update)

        # Update AIS table
        ais_update = db.session.query(Ais).filter(Ais.desig==ship_desig).all()
        for pos in ais_update:
            pos.ship_class = new_class
            db.session.add(pos)
        
        class_update.ship_class = new_class

        # Update mmsi table
        mmsi_update = db.session.query(Mmsi).filter(Mmsi.desig==ship_desig).all()
        for mmsi in mmsi_update:
            mmsi.ship_class = new_class
            db.session.add(mmsi)

        db.session.commit()

        flash(ship_desig + " ship class updated to " + new_class)

    
    # Create and validate forms to update the class sizes and values
    known_class_form = ChangeShipClassForm()

    if "submit_known_class" in request.form and known_class_form.validate_on_submit():
        ship_desig = known_class_form.desig_field.data
        new_class = known_class_form.class_select_field.data

        # Update ship class table
        class_update = db.session.query(Ship_Classes).filter(Ship_Classes.desig==ship_desig).first()
        class_update.ship_class = new_class

        db.session.add(class_update)

        # Update AIS table
        ais_update = db.session.query(Ais).filter(Ais.desig==ship_desig).all()
        for pos in ais_update:
            pos.ship_class = new_class
            db.session.add(pos)
        
        class_update.ship_class = new_class

        # Update mmsi table
        mmsi_update = db.session.query(Mmsi).filter(Mmsi.desig==ship_desig).all()
        for mmsi in mmsi_update:
            mmsi.ship_class = new_class
            db.session.add(mmsi)

        db.session.commit()

        flash(ship_desig + " ship class updated to " + new_class)

    return render_template('manage_data.html', class_form=class_form, table_script=script, \
                            table_div=div, class_script=class_script, class_div=class_div, \
                            known_desig_script=known_desig_script, known_desig_div=known_desig_div, \
                            known_class_form=known_class_form, title='Update Ship Class')

'''
# Data Stream endpoints
# AIS Data Stream

from pandas.errors import MergeError

old_ships = pd.DataFrame()
@app.route('/ais_stream', methods=['GET','POST'])
def ais_stream():
    global old_ships

    query_latitude = 36.712468 #36.46
    query_longitude = -122.770232 #-122.3

    # If this is the first query, conduct a query and return first ship positions, initialize old_ships
    if old_ships.empty:
        ships = ais_boundary_query(latitude = query_latitude, longitude = query_longitude, hours = 1, radius = 20)
        print("Ships\n")
        print(ships.columns)
        print("Old Ships\n")
        old_ships = ships
        print(old_ships.columns)

        new_positions = ships

        return jsonify(mmsi=new_positions['mmsi'].tolist(), imoNumber=new_positions['imoNumber'].tolist(), name=new_positions['name'].tolist(), 
        callSign=new_positions['callSign'].tolist(), cargo=new_positions['cargo'].tolist(), COG=new_positions['COG'].tolist(), 
        heading=new_positions['heading'].tolist(), navStatus=new_positions['navStatus'].tolist(), SOG=new_positions['SOG'].tolist(), 
        latitude=new_positions['latitude'].tolist(), longitude=new_positions['longitude'].tolist(), timeOfFix=new_positions['timeOfFix'].tolist(), 
        dist_from_sensor_km=new_positions['dist_from_sensor_km'].tolist(), dead_weight=new_positions['dead_weight'].tolist(), 
        length=new_positions['length'].tolist(), beam=new_positions['beam'].tolist(), desig=new_positions['desig'].tolist(),
        merc_latitude=new_positions['merc_latitude'].tolist(), merc_longitude=new_positions['merc_longitude'].tolist(),
        ship_class=new_positions['ship_class'].tolist())

    # Returns any new positions that aren't in the input dataframe, empty if no new positions
    print("Old Ships Not First\n")
    print(old_ships.columns)
    new_positions = ais_compare(old_ships, query_latitude, query_longitude)

    # If there are new positions, update the old ships dataframe
    if not new_positions.empty:
        old_ships = new_positions
        try:
            old_ships = old_ships.drop('_merge')
        except KeyError:
            pass


    return jsonify(mmsi=new_positions['mmsi'].tolist(), imoNumber=new_positions['imoNumber'].tolist(), name=new_positions['name'].tolist(), 
    callSign=new_positions['callSign'].tolist(), cargo=new_positions['cargo'].tolist(), COG=new_positions['COG'].tolist(), 
    heading=new_positions['heading'].tolist(), navStatus=new_positions['navStatus'].tolist(), SOG=new_positions['SOG'].tolist(), 
    latitude=new_positions['latitude'].tolist(), longitude=new_positions['longitude'].tolist(), timeOfFix=new_positions['timeOfFix'].tolist(), 
    dist_from_sensor_km=new_positions['dist_from_sensor_km'].tolist(), dead_weight=new_positions['dead_weight'].tolist(), 
    length=new_positions['length'].tolist(), beam=new_positions['beam'].tolist(), desig=new_positions['desig'].tolist(),
    merc_latitude=new_positions['merc_latitude'].tolist(), merc_longitude=new_positions['merc_longitude'].tolist(),
    ship_class=new_positions['ship_class'].tolist())



import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import matplotlib
matplotlib.use('Agg')

@app.route('/spectrogram_plot', methods=['GET', 'POST'])
def spectrogram_plot():
    #wsl
    #tgt_dir = "/home/lemgog/thesis/acoustic-inference-application/data/mars_regression_mar_aug_2019"
    #filename = "classB-278.56-9.51-_190601_10_2580.wav"
    #cluster
    tgt_dir = "/h/nicholas.villemez/thesis/acoustic-inference-application/data/single_testdata"
    filename = "classA_130321_16_60.wav"
    rate = 4000
    n_fft = 1024
    overlap = 50
    fName = os.path.splitext(filename)[0]
    fig, ax = plt.subplots(1, figsize=(10,10))

    signal, sr = librosa.load(os.path.join(tgt_dir, filename), sr=rate)

    # using matplotlib
    Pxx, freqs, bins, im = ax.specgram(signal, NFFT=n_fft, Fs = sr, noverlap=overlap)
    #plt.axis(ymax=300)

    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.title(filename + ' Spectrogram')
    #plt.colorbar(format='%+2.0f dB')
    save_name = os.path.join(tgt_dir, fName + ' spectro_matplotlib.png')
    plt.savefig(save_name)
    plt.close()

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    return Response(output.getvalue(), mimetype='image/png')
    #return send_file(save_name, mimetype='image/gif')
    #https://stackoverflow.com/questions/50728328/python-how-to-show-matplotlib-in-flask
'''


