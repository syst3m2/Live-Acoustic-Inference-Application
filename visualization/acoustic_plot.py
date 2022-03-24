from bokeh.layouts import column, gridplot

from bokeh.plotting import figure as bokfig
from bokeh.models.mappers import LogColorMapper
from scipy.signal import fir_filter_design, spectrogram
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, CustomJS, Div, Select
from bokeh.embed import components
import pandas as pd
import numpy as np
from bokeh.palettes import Viridis256, Inferno256
from bokeh.models import HoverTool, DatetimeTickFormatter, LabelSet, DataTable, TableColumn, PrintfTickFormatter, Text
from vs_data_query import WavCrawler
import json
from bokeh.events import ButtonClick
from bokeh.models import Button, CustomJS

# For ais
from bokeh.embed import server_document
from bokeh.embed import server_session
from pyproj import Proj, transform, Transformer
from bokeh.plotting import figure, curdoc
from bokeh.tile_providers import CARTODBPOSITRON, get_provider
from bokeh.embed import components
from bokeh.models import ColumnDataSource, Circle, HoverTool, MultiLine, Legend, LegendItem, Line, LabelSet

from bokeh.palettes import Spectral6, Category10_10
from bokeh.transform import factor_cmap
import pandas as pd
import os
from bokeh.models import WMTSTileSource
from bokeh.layouts import column, layout
import datetime
from bokeh.models.mappers import CategoricalColorMapper
import sys
sys.path.append(".")
sys.path.append("../..")
sys.path.append("..")
#from app import app, db
import sqlalchemy
from bokeh.io import curdoc, show
from bokeh.io import export_png
from math import pi

import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

ACOUSTIC_DATABASE = os.environ.get('ACOUSTIC_DATABASE_URL')
ACOUSTIC_DATABASE_FILE = os.environ.get('ACOUSTIC_DATABASE_URL')[10:]

def get_model_info(model_id, attribute, connection):
    query = "SELECT " + str(attribute) + " from models WHERE id="+str(model_id)
    response = connection.execute(query).first()[0]
    return response

def latlong_to_mercator(latitude, longitude):
    transformer = Transformer.from_crs('epsg:4326','epsg:3857', always_xy=True)
    merc_longitude, merc_latitude = transformer.transform(longitude, latitude)
    return merc_latitude, merc_longitude

# Use to check the range for AIS true labels
def check_range(t1, t2, radius, mmsi, connection):
    query = 'SELECT * FROM AIS \
        WHERE "timeOfFix" BETWEEN ' + str(t1) + ' AND ' + str(t2) + \
        ' AND dist_from_sensor_km >' + str(radius) + ' AND mmsi=' + str(mmsi)

    check_range = pd.read_sql_query(query, connection)

    if check_range.empty:
        return True
    else:
        return False

def get_class(mmsi, connection):
    query = "SELECT desig FROM mmsi WHERE mmsi=" + str(mmsi)
    desig = connection.execute(query).first()[0]
    class_query = "SELECT ship_class from ship_classes WHERE desig='" + desig +"'"
    ship_class = connection.execute(class_query).first()
    if ship_class is not None:
        ship_class = ship_class[0]
    else:
        ship_class = "Unknown"
    return ship_class

# Returns a JSON of true labels for a given time period
def get_true_labels(t1, t2, radius, connection):
    # t1 is 1 hour before the start time and t2 is desired time period after t1
    # which is 30 seconds in this case
    # Radius to evaluate predictions on is 30 kilometers
    
    label_t1 = int(t1) - (60*60)#(60*60*24)
    label_t2 = t2
    
    #radius = 20

    # Select the latest time where the radius is in the defined prediction radius within last 24 hours
    # until t2
    label_query = 'SELECT mmsi, MAX("timeOfFix") FROM AIS \
    WHERE "timeOfFix" BETWEEN ' + str(label_t1) + ' AND ' + str(label_t2) + ' AND dist_from_sensor_km <=' + str(radius) \
    + ' GROUP BY mmsi'

    true_labels = pd.read_sql_query(label_query, connection)
    true_labels['ship_class'] = true_labels.apply(lambda x: get_class(x['mmsi'], connection), axis=1)

    # perform another query between latest time for that ship and t1, if ship left radius before prediction, then not a true label
    true_labels = true_labels[true_labels.apply(lambda x: check_range(x['max'], t1, radius, x['mmsi'], connection), axis=1)]['ship_class']

    true_labels.name = "True Labels"

    true_labels = true_labels.to_json(orient='records')

    if len(json.loads(true_labels)) == 0:
        true_labels = '["Class E"]'

    return true_labels

#-----------------------------------Bokeh Plot Code--------------------------------------------
#def spectrogram_plot(doc):
sample_rate = 8000
n_fft = 1024
overlap = 25
PRED_FLAG = True

# Create a wavcrawler defined by the last batch of predictions, which is likely 30 minutes
file  = os.environ.get('ACOUSTIC_DATABASE_URL')[10:] #or '/home/lemgog/thesis/acoustic_app/data/mbari/master_index.db'

acoustic_db = os.environ.get('ACOUSTIC_DATABASE_URL') #or 'sqlite:////home/lemgog/thesis/acoustic_app/data/mbari/master_index.db'

# Query the latest time from the predictions database, then use that time to query the
# corresponding audio data
acoustic_engine = sqlalchemy.create_engine(acoustic_db)
acoustic_connection = acoustic_engine.connect()

app_db = os.environ.get('DATABASE_URL') #or 'sqlite:///../app.db'

app_engine = sqlalchemy.create_engine(app_db)
app_connection = app_engine.connect()

latest_time_query = "SELECT MAX(end_time) FROM PREDICTIONS"
t2 = app_connection.execute(latest_time_query)

#latest_time_query = "SELECT MAX(end_time_sec) FROM AUDIO"
#t2 = acoustic_connection.execute(latest_time_query)

t2 = t2.all()[0][0]

if t2 is None:
    print("No predictions in database")
    latest_time_query = "SELECT MAX(end_time_sec) FROM AUDIO"
    t2 = acoustic_connection.execute(latest_time_query)
    PRED_FLAG = False
    t2 = t2.all()[0][0]

t2 = int(t2)

t1 = t2-1800

#acoustic_connection.close()
#acoustic_engine.dispose()

wc = WavCrawler(file, t1, t2, segment_length=8000, overlap=0.25)

segment = next(wc)

signal = segment.samples[0, :]

f, t, Sxx = spectrogram(signal, sample_rate)
i=0
df_length = f.shape[0] * t.shape[0]
df_spectrogram = pd.DataFrame(np.nan, index=range(0,df_length), columns=['Frequency', 'Time', 'Sxx'])
for freq in range(f.shape[0]):
    for time in range(t.shape[0]):
        df_spectrogram.loc[i] = [f[freq],t[time],Sxx[freq][time]]
        i = i+1

df_spectrogram['Time'] = df_spectrogram['Time'] + float(segment.time_stamp)
df_spectrogram['Time'] = pd.to_datetime(df_spectrogram['Time'], unit='s')

TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"
PALETTE = Inferno256
mapper = LogColorMapper(palette=PALETTE, low=np.min(Sxx), high=np.max(Sxx), nan_color='black')
spectrogram_plot = bokfig(title="Spectrogram",x_axis_location="below", x_axis_type='datetime', plot_width=650, plot_height=400, tools=TOOLS)
spectrogram_plot.background_fill_color = "#eaeaea"
spectrogram_source = ColumnDataSource(data=df_spectrogram)
spec_circle = spectrogram_plot.circle(x="Time", y="Frequency", source=spectrogram_source, fill_color={'field': 'Sxx', 'transform': mapper}, line_color=None)

spec_hover = HoverTool(renderers=[spec_circle], tooltips = [("Frequency (Hz)", "@Frequency"), ('Time (s)', '@Time{"%A, %B %d %H:%M:%S"}'), ('Spectrogram', '@Sxx')], formatters={'@Time':'datetime'})

spectrogram_plot.add_tools(spec_hover)


color_bar = ColorBar(color_mapper=mapper,
                    label_standoff=4, border_line_color=None, width=8, location=(0, 0))
spectrogram_plot.add_layout(color_bar, 'right')

# Label styling
spectrogram_plot.xaxis.axis_label = "Time (seconds)"
spectrogram_plot.yaxis.axis_label = "Frequency (Hz)"
spectrogram_plot.title.align = 'center'

spectrogram_plot.xaxis.formatter = DatetimeTickFormatter(years="%m/%d/%Y %H:%M:%S",
                                          months="%m/%d/%Y %H:%M:%S",
                                          days="%m/%d/%Y %H:%M:%S",
                                          hours="%m/%d/%Y %H:%M:%S",
                                          hourmin="%m/%d/%Y %H:%M:%S",
                                          minutes="%m/%d/%Y %H:%M:%S",
                                          minsec="%m/%d/%Y %H:%M:%S",
                                          seconds="%m/%d/%Y %H:%M:%S",
                                          milliseconds="%m/%d/%Y %H:%M:%S",
                                          microseconds="%m/%d/%Y %H:%M:%S")
spectrogram_plot.xaxis.major_label_orientation = pi/8 #pi/4
#spectrogram_plot.xaxis.formatter = PrintfTickFormatter()

'''
Do not need anymore
# Add text with the latest time
df_spectrogram_time = df_spectrogram.iloc[[df_spectrogram['Time'].idxmax()]]

df_spectrogram_time = df_spectrogram_time.copy()

df_spectrogram_time.loc[:,'str_time'] = df_spectrogram_time.loc[:,'Time'].dt.strftime('%Y-%b-%d %H:%M')
#spectrogram_time = {"Current Time":df_spectrogram_time['Time'][0]}
text_source = ColumnDataSource(df_spectrogram_time)

#glyph = Text(x="Time", y="Frequency", text="str_time", x_offset=-300) #, text_color="#96deb3"
#spectrogram_plot.add_glyph(text_source, glyph)
glyph = LabelSet(x='Time', y='Frequency', text='str_time',
              x_offset=-300, source=text_source, render_mode='canvas', background_fill_color='white', background_fill_alpha=0.9, border_line_color='black', text_font_size='12pt')

spectrogram_plot.add_layout(glyph)
'''
# AIS stuff----------------------------------------------------------------------------------------

query_latitude = 36.712465
query_longitude = -122.187548
center_latitude=  36.712465
center_longitude= -122.187548

x_low = center_longitude + .5
y_low = center_latitude - .5
x_high = center_longitude - .5
y_high = center_latitude + .5

# Create columns with lat and longs in mercator format for plotting
new_y_low, new_x_low = latlong_to_mercator(y_low, x_low)
new_y_high, new_x_high = latlong_to_mercator(y_high, x_high)
new_center_latitude, new_center_longitude = latlong_to_mercator(center_latitude, center_longitude)

# Define data source for sensor location
sensor_location = ColumnDataSource(
        data=dict(
            lat=[new_center_latitude],
            lon=[new_center_longitude],
            centerlat = [center_latitude],
            centerlong = [center_longitude],
            name=['Acoustic Sensor Location']
            )
    )

#query = 'SELECT * FROM AIS WHERE "timeOfFix" > ' + str(t1) + ' AND "timeOfFix" < ' + str(t2)
query = 'SELECT * FROM AIS WHERE "timeOfFix" BETWEEN ' + str(t1) + ' AND ' + str(t2)

ships_df = pd.read_sql_query(query, app_engine)

ships_df['timeOfFix'] = pd.to_datetime(ships_df['timeOfFix'], unit='s')

source = ColumnDataSource(data=ships_df)


tile_provider = WMTSTileSource(
        url='https://server.arcgisonline.com/ArcGIS/rest/services/Ocean_Basemap/MapServer/tile/{z}/{y}/{x}',
        attribution='Tiles &copy; Esri &mdash; Sources: GEBCO, NOAA, CHS, OSU, UNH, CSUMB, National Geographic, DeLorme, NAVTEQ, and Esri'
    )


# Create basemap as plot
#tile_provider = get_provider(CARTODBPOSITRON)
ais_plot = figure(title = "AIS Plot", x_range=(new_x_low, new_x_high), 
                y_range=(new_y_low, new_y_high), x_axis_type="mercator", 
                y_axis_type="mercator", plot_height=400, plot_width=500)
ais_plot.title.align = "center"
#ais_plot.add_tile(tile_provider)
ais_plot.add_tile(tile_provider)



PALETTE = Spectral6
#mapper = CategoricalColorMapper(palette=PALETTE, factors='ship_class')
# use for fill color {'field': 'ship_class', 'transform': mapper}
# Add ship positions to plot
#circle = ais_plot.circle(x="merc_longitude", y="merc_latitude", size=5, fill_color='blue', fill_alpha=0.8, source=source)

# Create restructured dataframe configured for lines
ships_grouped_df = ships_df.groupby('mmsi')

# Make color list, corresponding to each unique ship_class
color_dictionary = {'Class A':Category10_10[0], 'Class B':Category10_10[1], 'Class C':Category10_10[2], 'Class D':Category10_10[3], 'Class E':Category10_10[4], 'Unknown':Category10_10[5]}

colors_list = []
class_list = []
#lat_list = []
#lon_list = []
for key, data in ships_grouped_df:
    colors_list.append(color_dictionary[data['ship_class'].unique()[0]])
    class_list.append(data['ship_class'].unique()[0])
    #lat_list.append(data.loc[data['timeOfFix'].idxmax()]['merc_latitude'])
    #lon_list.append(data.loc[data['timeOfFix'].idxmax()]['merc_longitude'])


idx = ships_df.groupby(['mmsi'])['timeOfFix'].transform(max) == ships_df['timeOfFix']

circle_ship_df = ships_df[idx].copy()
circle_ship_df['color'] =  circle_ship_df['ship_class'].map(color_dictionary)

line_source = ColumnDataSource(dict(
    xs=[list(x[1]) for x in ships_grouped_df.merc_longitude],
    ys=[list(y[1]) for y in ships_grouped_df.merc_latitude],
    classes=class_list,
    color=colors_list
))

circle_source = ColumnDataSource(circle_ship_df)
'''
circle_source = ColumnDataSource(dict(
    merc_longitude=lon_list,
    merc_latitude=lat_list,
    classes=class_list,
    color=colors_list

))
'''
ais_plot.multi_line(xs='xs', ys='ys', legend_field="classes", source=line_source, line_color='color', line_width=3)
circle = ais_plot.circle(x="merc_longitude", y="merc_latitude", size=5, 
                        fill_color='color',line_color='black', line_width=1, fill_alpha=0.8, source=circle_source)


'''
line_glyph = MultiLine(xs="xs",
            ys="ys", line_width=2, line_color='color')
multi_line_render = ais_plot.add_glyph(line_source, line_glyph)
'''

'''
# Multi line attempt using lines
legend_items=[]
for key, data in ships_grouped_df:
    #glyph = Line(x=daily_accuracy_scores['date'].tolist(), y=daily_accuracy_scores[column[i]].tolist(), line_color=mypalette[i], line_width=6, line_alpha=0.6)
    glyph = Line(x='merc_longitude', y='merc_latitude', line_color=color_dictionary[data['ship_class'].unique()[0]], line_width=6, line_alpha=0.6)
    render = ais_plot.add_glyph(line_source, glyph)
    #line_renderers.append(render)
    legend_items.append(LegendItem(label=cols[i],renderers=[render]))
'''
'''
legend = Legend(items=[
    LegendItem(label="Class A", renderers=[multi_line_render], index=0),
    LegendItem(label="Class B", renderers=[multi_line_render], index=1),
    LegendItem(label="Class C", renderers=[multi_line_render], index=2),
    LegendItem(label="Class D", renderers=[multi_line_render], index=3),
    LegendItem(label="Class E", renderers=[multi_line_render], index=4)
])

legend_layout = ais_plot.add_layout(legend)
'''

labels = LabelSet(x='merc_longitude', y='merc_latitude', text='desig',
              x_offset=5, y_offset=5, source=circle_source, render_mode='canvas', text_font_size='8pt', background_fill_color='white', background_fill_alpha=.8,border_line_color='black')

ais_plot.add_layout(labels)

# Add hover tools for ships and sensor location to plot
ais_hover = HoverTool(renderers=[circle],
                        tooltips=[('Name', '@name'), ('MMSI', '@mmsi'), ('Latitude', '@latitude'), 
                        ('Longitude', '@longitude'),('Time', '@timeOfFix{"%A, %B %d %H:%M:%S"}'),('Type', '@desig'), 
                        ('Ship Class', '@ship_class'),('Distance from Sensor (km)','@dist_from_sensor_km'), 
                        ('Bearing from Sensor','@bearing_from_sensor')], formatters={'@timeOfFix':'datetime', # use 'datetime' formatter for '@date' field
})

ais_plot.add_tools(ais_hover)

sensor = ais_plot.circle_x(x="lon", y="lat", size=10, fill_color='red', line_color='black', line_width=1, fill_alpha=0.5, source=sensor_location)
sensor_labels = LabelSet(x='lon', y='lat', text='name',
              x_offset=5, y_offset=5, source=sensor_location, render_mode='canvas', text_font_size='8pt')

ais_plot.add_layout(sensor_labels)
sensor_hover = HoverTool(renderers=[sensor], tooltips=[('Sensor Location', '@centerlat, @centerlong')])
ais_plot.add_tools(sensor_hover)



#---------------------------------END AIS Plot----------------------------------



#----------------------------Model Predictions-----------------------------------
query = "SELECT * FROM PREDICTIONS WHERE start_time <= " + str(t1) + " AND end_time >=" + str(t1)
predictions = pd.read_sql_query(query, app_engine)
#predictions['start_time'] = pd.to_datetime(predictions['start_time'], unit='s')

#need to update for multiple rows of predictions maybe? Just grabs first row is ok
# when display is only 30 seconds. Keep in mind the 25% overlap though, which would have multiple
# predictions on the same 30 second time period
if not predictions.empty:
    pred_json = json.loads(predictions['model_predictions'][0])
    #pred_df = pd.DataFrame.from_dict(pred_json, orient='index', columns=['Model Prediction'])
    pred_df = pd.DataFrame(pred_json).T
    pred_df.index = pred_df.index.rename("Model ID")
    prediction_df = pred_df.sort_values(by=['Model ID'])
    prediction_df = prediction_df.reset_index()

    prediction_df['Model Name'] = prediction_df.apply(lambda x: get_model_info(x['Model ID'], 'model_name', app_connection),axis=1)
    prediction_df['Model Type'] = prediction_df.apply(lambda x: get_model_info(x['Model ID'], 'model_type', app_connection),axis=1)
    prediction_df['Channels'] = prediction_df.apply(lambda x: get_model_info(x['Model ID'], 'channels', app_connection),axis=1)
    prediction_df['Model Choice'] = prediction_df.apply(lambda x: get_model_info(x['Model ID'], 'model_choice', app_connection),axis=1)

else:
    prediction_df = pd.DataFrame(data={"Model ID":[None],"Model Name":[None],"Model Type":[None],"Channels":[None],\
                                        "pred":[None],"pred_max_p":[None],"pred_vi_mean_max":[None],"entropy":[None],"nll":[None],"pred_std":[None],"var":[None],\
                                        "norm_entropy":[None],"epistemic":[None],"aleatoric":[None]})

prediction_df['color'] =  prediction_df['pred'].map(color_dictionary)

prediction_source = ColumnDataSource(data=prediction_df)

columns = [
        TableColumn(field = "Model ID", title="Model ID"),
        TableColumn(field = "Model Name", title="Model Name"),
        TableColumn(field = "Model Type", title="Model Type"),
        TableColumn(field = "Channels", title="Channels"),
        TableColumn(field = "pred", title="Model Prediction")
        #TableColumn(field = "pred_max_p", title="Pred Max P"),
        #TableColumn(field = "pred_vi_mean_max", title="Pred VI Mean Max"),
        #TableColumn(field = "entropy", title="Entropy"),
        #TableColumn(field = "nll", title="NLL"),
        #TableColumn(field = "pred_std", title="Pred STD"),
        #TableColumn(field = "var", title="Var"),
        #TableColumn(field = "norm_entropy", title="Norm Entropy"),
        #TableColumn(field = "epistemic", title="Epistemic"),
        #TableColumn(field = "aleatoric", title="Aleatoric")
]
prediction_table = DataTable(source=prediction_source, columns=columns, index_position=None, autosize_mode='fit_viewport') #width=400, height=280


# Create bar plot of model uncertainty

# Initializing the plot

prob_pred_df = prediction_df[prediction_df['Model Choice']=='dev_bnn_model']
#print(prob_pred_df)

#prob_pred_df.loc[:,'epistemic'] = prob_pred_df['epistemic'].astype('float')
#print(prob_pred_df.dtypes)

prob_pred_df.loc[:,'top'] =  prob_pred_df.loc[:,'epistemic']

prob_pred_source = ColumnDataSource(data=prob_pred_df)

#uncertainty = prediction_df[prediction_df['Model Choice']=='dev_bnn_model']
#model_name = prediction_df[prediction_df['Model Choice']=='dev_bnn_model']['Model Name']
#y_range=(0,1),
uncertainty_plot = figure(plot_height=400, x_range=prob_pred_source.data['Model ID'], title="Probabilistic Model Predictions and Uncertainty Values")

#print(prob_pred_df['epistemic'])
#Plotting
uncertainty_plot.vbar(x='Model ID',
                    top='top',
                    width=1,
                    fill_color='color',
                    source=prob_pred_source,
                    legend_field="pred",
                    line_color='black'
                    )
uncertainty_plot.xaxis.axis_label="Model IDs"
uncertainty_plot.yaxis.axis_label="Uncertainty"

uncert_labels = LabelSet(x='Model ID', y='top', text='pred', level='glyph',
        x_offset=-13.5, y_offset=0, source=prob_pred_source, render_mode='canvas')

uncertainty_plot.add_layout(uncert_labels)



# Create dropdown menu to change uncertainty metric

#uncert_callback = CustomJS(args=dict(src=prob_pred_source, uncertainty_plot=uncertainty_plot), code='''
#    console.log(' changed selected time', cb_obj.value);
#    var data = src.data;
#    data['top'] = data[cb_obj.value];
#    src.change.emit()

#    //console.log(uncertainty_plot.glyph.top)
#    //uncertainty_plot.glyph.top = {field: cb_obj.value};
#''')


def uncert_callback(attr, old, new):
    #view.filters = [GroupFilter(column_name='loc',group=select.value)]
    new_uncert = uncert_select.value
    prob_pred_source.data["top"] = prob_pred_source.data[new_uncert]

uncert_select = Select(options=['epistemic','aleatoric', 'pred_max_p','pred_vi_mean_max','entropy','nll','pred_std','var','norm_entropy'],value='epistemic', title = 'Uncertainty Metric')
uncert_select.on_change('value', uncert_callback)




'''
filter = GroupFilter(column_name='loc',group='Delhi')
view = CDSView(source=prob_pred_source, filters = [filter])
'''

#uncert_select.js_on_change('value', uncert_callback)
    




#----------------------End Predictions-----------------------------------------

#----------------------make table of true labels------------------------------

radius = 40

# If true labels are already in database
#true_labels = json.loads(predictions.iloc[0]['true_label'])[str(radius)].strip('][').split(',')
#pred_labels = {"True Labels":true_labels}

# If true labels need to be collected in real time
#if not predictions.empty:
#    predictions['ais_label'] = predictions.apply(lambda x: get_true_labels(x['start_time'], x['end_time'], radius, app_connection), axis=1)
#    pred_labels = {"AIS Labels":predictions.iloc[0]['ais_label'].strip('][').split(', ')}
#else:
#    pred_labels = {"AIS Labels": [None]}

#predictions['ais_label'] = 'test'
#pred_labels = {"AIS Labels":json.loads(predictions.iloc[0]['ais_label'])}
#pred_labels = {"AIS Labels":predictions.iloc[0]['ais_label'].split(',')}

classes = []
for mmsi, group in ships_grouped_df:
    classes.append(group['ship_class'].iloc[0])

pred_labels = {"AIS Labels":classes}

label_source = ColumnDataSource(data=pred_labels)
columns = [
        TableColumn(field = "AIS Labels", title="AIS Labels " + str(radius) + "km"),
]
label_table = DataTable(source=label_source, columns=columns, index_position=None, sizing_mode='stretch_width')#autosize_mode='fit_viewport'
# How much data to display in spectrogram, seconds times minutes, then multiply by sample rate

app_connection.close()
app_engine.dispose()

num_sec = 22
rollover = 185127 #9256350000 #9256350 #185127 #num_sec*8000
def update_spectrogram():
    # try to grab the next segment, if at the end of the wc object, then create a new one with new times

    try:
        segment = next(wc)
    except StopIteration:
        print("Stop iteration reached")
        #Change this to query the latest time from the predictions database, then user that time to query the
        #wavcrawler
        # compare old time to new time, if no difference, then return, continue checking but don't update
        # spectrogram data if no new data
        
        app_db = os.environ.get('DATABASE_URL') #or 'sqlite:///../app.db'

        app_engine = sqlalchemy.create_engine(app_db)
        app_connection = app_engine.connect()

        latest_time_query = "SELECT MAX(end_time) FROM PREDICTIONS"
        t2 = app_connection.execute(latest_time_query)

        #latest_time_query = "SELECT MAX(end_time_sec) FROM AUDIO"
        #t2 = acoustic_connection.execute(latest_time_query)

        t2 = int(t2.all()[0][0])
        t1 = t2-1800

        app_connection.close()
        app_engine.dispose()

        # One option for updating t1 and t2
        #t1 = int(int(segment.time_stamp) + dt.item().total_seconds())
        #t2 = int(t1 + 30)
        # Other option is to check file for new values with predictions, 
        #time_read = open('/home/lemgog/thesis/acoustic_app/model_predictor/time_sync.json', 'r')
        #time_read = json.loads(time_read.read())
        #t1 = time_read['start_time']
        #t2 = time_read['end_time']
        # if no new predictions, then continue without updating data
        # check if the current segments first timestamp is greater than the t1 of the new timestamp from the predictions
        # If it is not, then that means there are new predictions, so create a new wavcrawler
        # If it is, there are no new predictions, and the plot will stay paused at current values
        if int(spectrogram_source.data['Time'][0]) >= t1:
            print("No new predictions available, waiting...")
            return
        print("Creating new Wavcrawler object for new predictions")
        new_wc = WavCrawler(file,t1, t2, segment_length=8000, overlap=0.25)
        segment = next(new_wc)

    
    signal = segment.samples[0, :]
    f, t, Sxx = spectrogram(signal, sample_rate)
    i=0
    df_length = f.shape[0] * t.shape[0]
    new_df_spectrogram = pd.DataFrame(np.nan, index=range(0,df_length), columns=['Frequency', 'Time', 'Sxx'])
    for freq in range(f.shape[0]):
        for time in range(t.shape[0]):
            new_df_spectrogram.loc[i] = [f[freq],t[time],Sxx[freq][time]]
            i = i+1

    new_df_spectrogram['Time'] = new_df_spectrogram['Time'] + float(segment.time_stamp)
    new_df_spectrogram['Time'] = pd.to_datetime(new_df_spectrogram['Time'], unit='s')

    # Data to keep in frame, should be desired number of seconds * 8000 (sample rate)

    spectrogram_source.stream(new_df_spectrogram, rollover=rollover)

    #new_df_spectrogram_time = new_df_spectrogram.iloc[[new_df_spectrogram['Time'].idxmax()]]
    #new_df_spectrogram_time = new_df_spectrogram_time.copy()
    
    #new_df_spectrogram_time.loc[:,'str_time'] = new_df_spectrogram_time.loc[:,'Time'].dt.strftime('%Y-%b-%d %H:%M')
    #text_source.data = ColumnDataSource.from_df(new_df_spectrogram_time)

    t1 = int(segment.time_stamp)
    
    app_db = os.environ.get('DATABASE_URL') #or 'sqlite:///../app.db'
    app_engine = sqlalchemy.create_engine(app_db)
    app_connection = app_engine.connect()
    
    
    #---------------------Update model predictions--------------------------------------
    query = "SELECT * FROM PREDICTIONS WHERE START_TIME <= " + str(t1) + " AND END_TIME >=" + str(t1)
    predictions = pd.read_sql_query(query, app_engine)
    #predictions['start_time'] = pd.to_datetime(predictions['start_time'], unit='s')
    
    if not predictions.empty:
        pred_json = json.loads(predictions['model_predictions'][0])
        #pred_df = pd.DataFrame.from_dict(pred_json, orient='index', columns=['Model Prediction'])
        pred_df = pd.DataFrame(pred_json).T
        pred_df.index = pred_df.index.rename("Model ID")
        prediction_df = pred_df.sort_values(by=['Model ID'])
        prediction_df = prediction_df.reset_index()
        prediction_df['Model Name'] = prediction_df.apply(lambda x: get_model_info(x['Model ID'], 'model_name', app_connection),axis=1)
        prediction_df['Model Type'] = prediction_df.apply(lambda x: get_model_info(x['Model ID'], 'model_type', app_connection),axis=1)
        prediction_df['Channels'] = prediction_df.apply(lambda x: get_model_info(x['Model ID'], 'channels', app_connection),axis=1)
        prediction_df['Model Choice'] = prediction_df.apply(lambda x: get_model_info(x['Model ID'], 'model_choice', app_connection),axis=1)

    else:
        print("No predictions for this time")
        prediction_df = pd.DataFrame(data={"Model ID":[None],"Model Name":[None],"Model Type":[None],"Channels":[None],\
                                        "pred":[None],"pred_max_p":[None],"pred_vi_mean_max":[None],"entropy":[None],"nll":[None],"pred_std":[None],"var":[None],\
                                        "norm_entropy":[None],"epistemic":[None],"aleatoric":[None]})

    prediction_source.data = prediction_df

    prediction_df['color'] =  prediction_df['pred'].map(color_dictionary)

    prob_pred_df = prediction_df[prediction_df['Model Choice']=='dev_bnn_model']
    #prob_pred_df.loc[:,'epistemic'] = prob_pred_df['epistemic'].astype('float')
    #print(prob_pred_df['epistemic'])
    #print(prob_pred_df.dtypes)

    new_uncert = uncert_select.value
    #print(uncert_select.value)
    #new_top = prob_pred_df.loc[:,new_uncert] #.copy()
    prob_pred_df.loc[:,"top"] = prob_pred_df.loc[:,new_uncert]
    prob_pred_source.data = prob_pred_df

    # If true labels are already in database
    #true_labels = json.loads(predictions.iloc[0]['true_label'])[str(radius)].strip('][').split(', ')
    #pred_labels = {"True Labels":true_labels}

    # If true labels need to be collected in real time
    #predictions['ais_label'] = predictions.apply(lambda x: get_true_labels(x['start_time'], x['end_time'], radius, app_connection), axis=1)
    #pred_labels = {"AIS Labels":predictions.iloc[0]['ais_label'].strip('][').split(', ')}

    #classes = new_ship_pos.apply(lambda x: get_class(x['mmsi'], app_connection)
    #label_source.data = pred_labels

    #-----------------------Update AIS information--------------------------------
    #Update logic to include latest time within range from true labels df

    # UPDATE TO DELETE SHIPS THAT HAVE LEFT RANGE, GROUPBY MMSI, AND DELETE ONES THAT AREN'T IN RANGE

    t1_ais = t1 - (60*60*1) #(60*60*24)
    t2 = t1 + 1
    radius = 40

    query = 'SELECT * FROM AIS WHERE "timeOfFix" >= ' + str(t1_ais) + ' AND "timeOfFix" <= ' + str(t2) + \
            ' AND dist_from_sensor_km <=' + str(radius)

    new_ship_pos = pd.read_sql_query(query, app_engine)

    if new_ship_pos.empty:
        print("No ship positions for this time, check AIS stream")

    # Check if any ship exited, delete if they did
    allowed_mmsis = new_ship_pos.groupby('mmsi').agg({'timeOfFix':'max'})

    allowed_mmsis = allowed_mmsis.reset_index()

    allowed_mmsis = allowed_mmsis[allowed_mmsis.apply(lambda x: check_range(x['timeOfFix'], t1, radius, x['mmsi'], app_connection), axis=1)]['mmsi']

    new_ship_pos = new_ship_pos[new_ship_pos['mmsi'].isin(allowed_mmsis)]

    new_ship_pos['timeOfFix'] = pd.to_datetime(new_ship_pos['timeOfFix'], unit='s')
    
    source.data = ColumnDataSource.from_df(new_ship_pos)

    # Update source for lines
    new_ships_grouped_df = new_ship_pos.groupby('mmsi')
    colors_list = []
    class_list = []
    #lat_list=[]
    #lon_list=[]
    for key, data in new_ships_grouped_df:
        colors_list.append(color_dictionary[data['ship_class'].unique()[0]])
        class_list.append(data['ship_class'].unique()[0])
        #lat_list.append(data.loc[data['timeOfFix'].idxmax()]['merc_latitude'])
        #lon_list.append(data.loc[data['timeOfFix'].idxmax()]['merc_longitude'])

    new_ships_data = dict(
                    xs=[list(x[1]) for x in new_ships_grouped_df.merc_longitude],
                    ys=[list(y[1]) for y in new_ships_grouped_df.merc_latitude],
                    classes=class_list,
                    color=colors_list
                    )

    idx = new_ship_pos.groupby(['mmsi'])['timeOfFix'].transform(max) == new_ship_pos['timeOfFix']

    new_circle_ship_df = new_ship_pos[idx].copy()
    new_circle_ship_df['color'] =  new_circle_ship_df['ship_class'].map(color_dictionary)

    line_source.data = new_ships_data
    #circle_source.data = new_circle_source
    circle_source.data = ColumnDataSource.from_df(new_circle_ship_df)

    # Update AIS labels
    #classes = new_ships_grouped_df.apply(lambda x: get_class(x['mmsi'], app_connection), axis=1)
    #for mmsi, group in new_ships_grouped_df.
    

    # Use for if grabbing ship classes from database
    #mmsis = new_ship_pos['mmsi'].unique().tolist()
    #classes = []
    #if len(mmsis) > 0:
    #    for mmsi in mmsis:
    #        classes.append(get_class(mmsi, app_connection))
    #else:
    #    classes = ["None"]

    classes = []
    for mmsi, group in new_ships_grouped_df:
        classes.append(group['ship_class'].iloc[0])

    pred_labels = {"AIS Labels":classes}

    label_source.data = pred_labels

    app_connection.close()
    app_engine.dispose()

    return

for i in range(10):
    update_spectrogram()

ais_div = Div(text="""<b>AIS Classes Present</b>""", sizing_mode='stretch_width', align='center')
model_div = Div(text="""<b>Predicted Classes and Uncertainty</b>""", sizing_mode='stretch_width', align='center')

curdoc().add_root(gridplot([[ais_plot, spectrogram_plot],[ais_div,model_div],[label_table, prediction_table],[uncert_select,uncertainty_plot]], merge_tools=False))
#curdoc().add_root(layout([[ais_plot], [label_table], [prediction_table],[spectrogram_plot]], sizing_mode="stretch_width")) #
curdoc().add_periodic_callback(update_spectrogram, 1000)