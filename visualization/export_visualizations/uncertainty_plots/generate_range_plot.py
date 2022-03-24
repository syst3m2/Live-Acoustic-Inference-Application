# Generate range plot for single ship

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter, HourLocator
import os
import datetime
import sys
import argparse

from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import accuracy_score
from math import log
from matplotlib import cm
import sqlalchemy as db
import json

"""
Program to plot range vs time and classifications
Assumes that class labels are integers
Args:
    tgt_file: Ground truth data from AIS
    results_file: Prediction results of model classification
    plt_title: Title for plot
    save_name: File name for plot ouput file, can include path
    max_rng: Max value for y axis
    num_classes: Number of different classes predicted total
    start_hr: Integer start hr, for x axis bounds
    start_min: Integer start min, for x axis bounds
    end_hr: Integer end hr, for x axis bounds
    end_min: Integer end min, for x axis bounds
    uncertainty: label as true if csv contains entropy and std data
"""
#---------------------------update variables to run script-------------------------------------------------
#tgt_file = '/h/teams/visualizations/visualization_projects/0804_mars_viz/step3_trackcsv/mars_single_ais.csv'


#results_file = '/h/teams/visualizations/visualization_projects/0804_mars_viz/step5_models/dropout_1channel/multiclas_2021-08-20_14-24-25/verified_samples.csv'
#save_name = '/h/teams/visualizations/visualization_projects/0804_mars_viz/step6_output/dropout_1channel/mars_single_range_plot_1_channel_entropy_v3.png'

#results_file = '/h/teams/visualizations/visualization_projects/0804_mars_viz/step5_models/dropout_4channel/multiclas_2021-08-20_14-18-13/verified_samples.csv'


#save_name_eps = '/h/teams/visualizations/visualization_projects/mars_viz_07_14_2019/step6_output/dropout_1channel/mars_single_range_plot_probabilistic_dropout_1_channel_entropy.eps'
#save_name_pdf = '/h/teams/visualizations/visualization_projects/mars_viz_07_14_2019/step6_output/dropout_1channel/mars_single_range_plot_probabilistic_dropout_1_channel_entropy.pdf'
true_labels = '/h/teams/visualizations/visualization_projects/0804_mars_viz/step4_audio/test_labels.csv'
uncertainty = True
circle_size = 'entropy' #select uncertainty metric to use for circle sizes
orders_magnitude = 1000 # define what to multiply uncertainty values by for circle size
#-----------------------------------------------------------------------------------------------------------

def json_model_data(predictions):
    pred_json = json.loads(predictions)
    pred_df = pd.DataFrame(pred_json)
    pred_df=pred_df.reset_index(drop=True)

    pred_df['Category'] = ['class', 'pred_max_p', 'pred_vi_mean_max', 'entropy','nll','pred_std','var','norm_entropy','epistemic','aleatoric']
    #pred_df = pd.DataFrame(pred_df.iloc[0]).T
    #pred_df = pd.DataFrame(pred_df).T
    return pred_df

app_db = os.environ.get('DATABASE_URL')

app_engine = db.create_engine(app_db)
app_connection = app_engine.connect()

# Grab data from database and format
query = "SELECT * FROM PREDICTIONS WHERE id>=(SELECT MAX(id) FROM PREDICTIONS)-10" #"SELECT * FROM PREDICTIONS WHERE true_label IS NOT NULL" #"SELECT * FROM PREDICTIONS"

#SELECT * FROM `PREDICTIONS` WHERE true_label IS NOT NULL AND TRIM(true_label) <> ''

predictions = pd.read_sql_query(query, app_connection)


# Need to repeat this process for every radius, e.g. for x in radius_list
radius=40

accuracy_df = pd.DataFrame()
for index, row in predictions.iterrows():
    models = json_model_data(row['model_predictions'])

    '''
    # Skip rows where there is no true label because ship is in the ambiguous zone between range and no ship identification
    if None in set(json.loads(row['true_label'])[str(radius)]):
        #print(set(json.loads(row['true_label'])[str(radius)]))
        continue
    else:
        #print(set(json.loads(row['true_label'])[str(radius)]))
        models['true_label_multi_label'] = [set(json.loads(row['true_label'])[str(radius)])]
    '''        

    start_time = row['start_time']
    end_time = row['end_time']

    models['start_time'] = start_time
    models['end_time'] = end_time
    
    #tmp_accuracy = models.append(append_data)
    '''
    if accuracy_df.empty:
        accuracy_df = pd.DataFrame([models])
    else:
        accuracy_df = pd.concat([accuracy_df, pd.DataFrame([models])])
    '''
    accuracy_df = pd.concat([accuracy_df, models], ignore_index=True)

# Grab AIS data
# 
'''

# To find mmsi

mmsi_query = "SELECT mmsi, start_time, end_time FROM ais_times WHERE cpa<5 and radius=30"

mmsi_times = pd.read_sql_query(mmsi_query, app_connection)

mmsi_options_list = []
for index, row in mmsi_times.iterrows():



# For MMSI

ais_query = "SELECT * FROM AIS WHERE mmsi=" + str(mmsi) + " AND radius<=" + str(radius)

# Just for time
#ais_query = 'SELECT * FROM AIS WHERE "timeOfFix" BETWEEN ' + start_time + ' AND ' + end_time + ' AND radius<=' + str(radius)

ais_df = pd.read_sql_query(ais_query, app_connection)


# Cycle through columns, make plot for each model and save
cols = accuracy_df.columns
cols = list(cols)
cols.remove('start_time')
cols.remove('end_time')
cols.remove('Category')


accuracy_df['start_time'] = pd.to_datetime(accuracy_df['start_time'], unit='s')
accuracy_df['end_time'] = pd.to_datetime(accuracy_df['end_time'], unit='s')
ais_df['timeOfFix'] = pd.to_datetime(ais_df['timeOfFix'], unit='s')
accuracy_df['date'] = accuracy_df['start_time'].apply(lambda x: x.date())
accuracy_df['date'] = pd.to_datetime(accuracy_df['date'])


for column in cols:

    save_name = '/home/nicholas.villemez@ern.nps.edu/inference_application/dev_acoustic_app/visualization/export_visualizations/output/mars_range_plot_' + str(column) + '.png'

    # Need to drop NA values from each column, because each model wasn't necessarily activate the entire time

    new_accuracy_df = accuracy_df[accuracy_df[column].notna()]



    # Add grid



    max_rng = 40
    num_classes = 30
    start_hr = 0
    start_min = 0
    end_hr = 0
    end_min = 0
    true_label_class = 30

    results_x = []
    results_y = []


    # Initialize time data
    earliest_time = new_accuracy_df['start_time'].min() #pd.Timestamp(yr, mon, day, hr, minute, sec)
    latest_time = new_accuracy_df['start_time'].max() #pd.Timestamp(yr, mon, day, hr, minute, sec)

    for line in new_accuracy_df.iterrows():

        clip_time = line['start_time'] #pd.Timestamp(yr, mon, day, hr, minute, sec) # time of inference

        results_x.append(clip_time)
        # put the prediction on the closest AIS time available.
        results_idx = data['timeOfFix'].sub(clip_time).abs().idxmin() # results array is inference data and data is ais line plot

        # subtract the clip time from the ais time, grab the absolute value of the smallest value, get that index
        # This is the time difference absolute value
        time_diffs = data['TIME'].sub(clip_time).abs()

        #  we have the index of the smallest time difference, so now we will check to either side of that index
        # The smaller one is the one the point lies between

        timeplus = time_diffs[results_idx+1]
        timeminus = time_diffs[results_idx-1]

        smallest_time = min(timeplus, timeminus)

        # Get the index of this time
        results_idx_2 = time_diffs[time_diffs == smallest_time].index[0]

        #results_idx_2 = data[data['TIME'] == smallest_time]['TIME'].index[0]

        # Determine range difference between two times
        closest_range = data['RNG'][results_idx]
        second_range = data['RNG'][results_idx_2]

        closest_time = data['TIME'][results_idx]
        second_time = data['TIME'][results_idx_2]

        # Calculate the range difference per second

        seconds_diff = (closest_time - second_time).total_seconds()
        range_diff = abs(closest_range - second_range)

        km_per_sec = range_diff/seconds_diff

        pred_seconds_diff = (clip_time - closest_time).total_seconds()

        new_range_diff = pred_seconds_diff * km_per_sec

        range_diff_indicator = closest_range < second_range
        # This means that we must subtract a range value from the closest range
        if range_diff_indicator:
            new_range = closest_range - new_range_diff
        # Else we add to the range
        else:
            new_range = closest_range + new_range_diff

        results_y.append(new_range)

        #results_y.append(data['RNG'][results_idx])

        # new section ends

        #results_y.append(data['RNG'][results_idx])

        if clip_time > latest_time:
            latest_time = clip_time

            end_hr = hr
            
            if end_hr > 21:
                end_hr = end_hr - 22
                end_day = day + 1
            else:
                end_hr = hr + 1
                end_day = day

            end_min = 0

        if clip_time < earliest_time:
            earliest_time = clip_time
            start_hr = hr
            
            if start_hr < 2:
                start_hr = 24 + start_hr
                start_day = day - 1
            
            else:
                start_hr = hr
                start_day = day

            start_min = 0

    results_class = results_data['CLASS']

    true_predictions = pd.read_csv(true_labels,  header=0, names=['FILE_NAME','LABEL'])
    true_predictions = true_predictions['LABEL']

    prediction_dictionary = {'classA':0, 'classB':1, 'classC':2, 'classD':3, 'classE':4}
    true_predictions = [prediction_dictionary[x] for x in true_predictions]

    colors = []

    colors_for_map = {0:'red', 1:'yellow', 2:'orange', 3:'purple', 4:'green'}

    colors = [colors_for_map[x] for x in results_class]

    #better time formatting
    plt.rcParams.update({'font.size': 22})
    f = plt.figure(figsize=(14,14))
    # format x axis
    plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%m'))
    plt.gca().xaxis.set_major_locator(HourLocator(interval=1))
    plt.gca().xaxis.set_tick_params(rotation = 30)


    #f.set_rasterized(True) 
    #plt.gca().set_rasterized(True)
    #ax = f.add_subplot(111) 
    #ax.set_rasterized(True)

    #plt.title(plt_title)
    # plot range vs time
    plt.plot(x_data, y_data,c='black')

    # Use entropy or std to define the sizes of circle, multiply by orders of magnitude to translate to plot

    if uncertainty == True:
        sizes = results_data[circle_size] * orders_magnitude

        #sizes = sizes.to_list()
        sizes = sizes.to_numpy()

        # plot classification events
        #scatter = plt.scatter(results_x, results_y, alpha=.3, s=sizes, c='gray',label='Predicted', cmap=colormap, zorder=1)
        #scatter = plt.scatter(results_x, results_y, alpha=1, s=100, c=colors,label='Predicted', cmap=colormap, zorder=2)
        scatter = plt.scatter(results_x, results_y, alpha=.3, s=sizes, c='gray',label='Predicted', zorder=1)
        scatter = plt.scatter(results_x, results_y, alpha=1, s=100, c=colors,label='Predicted', zorder=2)
        

    else:
        scatter = plt.scatter(results_x, results_y, alpha=1, s=50, c=colors,label='Predicted')

    plt.ylim(0, max_rng)
    plt.xlim([datetime.datetime(yr, mon, start_day, start_hr, start_min, 0), datetime.datetime(yr, mon, end_day, end_hr, end_min, 0)])

    plt.grid()

    # original legend

    #plt.legend(*scatter.legend_elements(), loc="upper right", title="Classes") 

    plt.plot([],[], marker="o", ms=10, ls="")

    texts = ["Class A", "Class B", "Class C", "Class D", "Class E"]
    patches = [ plt.plot([],[], marker="o", ms=10, ls="", mec=None, color=colors_for_map[i], 
                label="{:s}".format(texts[i]) )[0]  for i in range(len(texts)) ]

    #if uncertainty==True:
    #    anchor = (0, .25)
    #else:
    anchor = (0,0)

    leg1 = plt.legend(handles=patches, loc='lower left', bbox_to_anchor = anchor, title='Predictions')
    plt.gca().add_artist(leg1)

    if uncertainty == True:
        # create second legend
        percentile_25 = np.percentile(sizes, 25)
        percentile_50 = np.percentile(sizes, 50)
        percentile_75 = np.percentile(sizes, 75)

        scatter1 = plt.scatter([], [], c='k', alpha=0.3, s=percentile_25, label='25th Percentile' )
        scatter2 = plt.scatter([], [], c='k', alpha=0.3, s=percentile_50, label='50th Percentile')
        scatter3 = plt.scatter([], [], c='k', alpha=0.3, s=percentile_75, label='75th Percentile')

        # find center of plot x axis
        time_delta = datetime.datetime(yr, mon, end_day, end_hr, end_min, 0) - datetime.datetime(yr, mon, start_day, start_hr, start_min, 0)
        #center_dist = time_delta/2
        #center = datetime.datetime(yr, mon, end_day, end_hr, end_min, 0)-center_dist

        leg2 = plt.legend(handles = [scatter1, scatter2, scatter3], scatterpoints=1, frameon=True, labelspacing=1.5, loc='lower right', borderpad=1)

    # add accuracy score to plot
    accuracy = accuracy_score(true_predictions, results_class)
    accuracy = round(accuracy, 4)
    accuracy = str(accuracy)
    #plt.text(datetime.datetime(yr, mon, start_day, start_hr, start_min+1, 0), 9, 'Accuracy: ' + accuracy, fontsize=20)
    save_name = save_name[:-4] + "_" + accuracy + save_name[-4:]
    #save_name_eps = save_name_eps[:-4] + "_" + accuracy + save_name_eps[-4:]

    plt.xlabel("Time")
    plt.ylabel("Range (km)")
    plt.savefig(save_name, dpi=300)


    #plt.savefig(save_name_eps, dpi=300)
    #plt.savefig(save_name_pdf, dpi=300)
    #plt.savefig(save_name_eps, rasterized=True, dpi=300)



    plt.close()
'''