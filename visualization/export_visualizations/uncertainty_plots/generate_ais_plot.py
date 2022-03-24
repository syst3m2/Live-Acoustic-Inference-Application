# Generate AIS plot for single ship

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score
from math import log

#---------------------------update variables to run script-------------------------------------------------
# tgt_file is the csv generated of the ais tracks from the .mat files
# Results file is the predictions from the model
# plot file is the output graph of AIS. 
# Center latitude and longitude are the location the graph centers around. Different for HARP and MARS data
tgt_file = '/h/teams/visualizations/visualization_projects/0719_mars_viz/step3_trackcsv/mars_single_ais.csv'


plot_file = '/h/teams/visualizations/visualization_projects/0804_mars_viz/step6_output/dropout_1channel/mars_single_ais_plot_1channel_entropy.png'
results_file = '/h/teams/visualizations/visualization_projects/0804_mars_viz/step5_models/dropout_1channel/multiclas_2021-08-19_15-41-03/verified_samples.csv'

#plot_file = '/h/teams/visualizations/visualization_projects/0719_mars_viz/step6_output/dropout_4channel/mars_single_ais_plot_4channel_entropy.png'
#results_file = '/h/teams/visualizations/visualization_projects/0719_mars_viz/step5_models/dropout_4channel/multiclas_2021-08-19_15-26-44/verified_samples.csv'



#plot_file_eps = '/h/teams/visualizations/visualization_projects/mars_viz_07_14_2019/step6_output/dropout_1channel/mars_single_ais_plot_1channel_entropy.eps'
true_labels = '/h/teams/visualizations/visualization_projects/0804_mars_viz/step4_audio/test_labels.csv'
# 36.712465 -122.187548
sensor_latitude = 36.712465 #36.712468 #32.666
sensor_longitude = -122.187548 #-122.770232 #-117.707
center_latitude=  36.712465 #36.712468 #32.666 #32.65
center_longitude= -122.187548 #-122.770232  #-117.707 #-117.7
#plot_title = 'AIS Plot - Deterministic'
uncertainty=True
circle_size = 'entropy' #select STD or ENTROPY to define the circle sizes
orders_magnitude = 1000
'''
if circle_size == 'ENTROPY':
    orders_magnitude = 2000 # define what to multiply uncertainty values by for circle size
elif circle_size == 'STD':
    orders_magnitude = 3000
'''
#----------------------------------------------------------------------------------------------------------


x_low = center_longitude + 1
y_low = center_latitude - .75
x_high = center_longitude - 1
y_high = center_latitude + .75

fig = plt.figure(figsize=(12,9))
#ax = fig.add_subplot(111)

m = Basemap(projection='mill',
            llcrnrlat = y_low,
            urcrnrlat = y_high,
            llcrnrlon = x_high,
            urcrnrlon = x_low,
            resolution = 'h'
            )

m.drawcoastlines()

m.drawparallels(np.arange(y_low,y_high,.2), labels=[True,False,False,False])
m.drawmeridians(np.arange(x_high,x_low,.2), labels=[0,0,0,1])

# grab data to plot
data = pd.read_csv(tgt_file, parse_dates=[0], header=0)
data = data.sort_values(by='TIME', ascending=True)

sites_lat_y = data.LAT.to_list()
sites_lon_x = data.LONG.to_list()

if uncertainty == True:
    results_data = pd.read_csv(results_file,  header=0, names=['FPATH', 'CLASS', 'entropy', 'std', 'pred_max_p_vi','pred_vi_mean_max_p','nll_vi','var','norm_entropy_vi','epistemic','aleatoric'])
else:
    results_data = pd.read_csv(results_file,  header=0, names=['FPATH', 'CLASS'])
results_x = []
results_y = []

for line in results_data.iterrows():
    yr = (int)('20' + line[1]['FPATH'].split('_')[-3][:2])
    day = (int)(line[1]['FPATH'].split('_')[-3][4:])
    mon = (int)(line[1]['FPATH'].split('_')[-3][2:4])
    hr = (int)(line[1]['FPATH'].split('_')[-2])
    temp = line[1]['FPATH'].split('_')[-1]
    time = (int)(temp.split('.')[0])
    minute = time // 60
    sec = time % 60
    clip_time = pd.Timestamp(yr, mon, day, hr, minute, sec) # time of inference

    # put the prediction on the closest AIS time available.
    results_idx = data['TIME'].sub(clip_time).abs().idxmin() # results array is inference data and data is ais line plot
    results_y.append(data['LAT'][results_idx])
    results_x.append(data['LONG'][results_idx])
    '''
    # new method
    time_diffs = data['TIME'].sub(clip_time).abs()

    #  we have the index of the smallest time difference, so now we will check to either side of that index
    # The smaller one is the one the point lies between

    timeplus = time_diffs[results_idx+1]
    timeminus = time_diffs[results_idx-1]

    smallest_time = min(timeplus, timeminus)

    # Get the index of this time
    results_idx_2 = time_diffs[time_diffs == smallest_time].index[0]

    #results_idx_2 = data[data['TIME'] == smallest_time]['TIME'].index[0]

    # Determine latitude/longitude difference between two times
    closest_lat = data['LAT'][results_idx]
    second_lat = data['LAT'][results_idx_2]

    closest_long = data['LONG'][results_idx]
    second_long = data['LONG'][results_idx_2]

    closest_time = data['TIME'][results_idx]
    second_time = data['TIME'][results_idx_2]

    # Calculate the lat/long difference per second

    seconds_diff = (closest_time - second_time).total_seconds()
    lat_diff = abs(closest_lat - second_lat)
    long_diff = abs(closest_long - second_long)

    lat_per_sec = lat_diff/seconds_diff
    long_per_sec = long_diff/seconds_diff

    pred_seconds_diff = (clip_time - closest_time).total_seconds()

    new_lat_diff = pred_seconds_diff * lat_per_sec
    new_long_diff = pred_seconds_diff * long_per_sec

    lat_diff_indicator = closest_lat < second_lat
    long_diff_indicator = closest_long < second_long
    # This means that we must subtract a range value from the closest range
    if lat_diff_indicator:
        new_lat = closest_lat - new_lat_diff
    # Else we add to the range
    else:
        new_lat = closest_lat + new_lat_diff

    results_y.append(new_lat)

    # This means that we must subtract a range value from the closest range
    if long_diff_indicator:
        new_long = closest_long - new_long_diff
    # Else we add to the range
    else:
        new_long = closest_long + new_long_diff

    results_x.append(new_long)
    '''

results_class = results_data['CLASS']

true_predictions = pd.read_csv(true_labels,  header=0, names=['FILE_NAME','LABEL'])
true_predictions = true_predictions['LABEL']

prediction_dictionary = {'classA':0, 'classB':1, 'classC':2, 'classD':3, 'classE':4}
true_predictions = [prediction_dictionary[x] for x in true_predictions]

colors = []
colors_for_map = {0:'red', 1:'yellow', 2:'orange', 3:'purple', 4:'green'}

colors = [colors_for_map[x] for x in results_class]

'''
index = 0
for pred in results_class:
    if pred == true_predictions[index]:
        colors.append(pred)
    else:
        colors.append(pred)
        #false_pred[class_num] += 1
    index += 1
colors_for_map = ['red', 'yellow', 'orange', 'purple', 'green']

colormap = ListedColormap(colors_for_map)

#colors = ['green', 'darkblue', 'yellow', 'red', 'blue', 'orange']
'''
# For multiple ships
# grouped_data = data.groupby('MMSI')

# s change size
#m.scatter(sites_lon_x, sites_lat_y, latlon=True, s=5000, c='red', marker='^', alpha=1, edgecolor='k', linewidth=1, zorder=2)

if uncertainty == True:
    # Use entropy or std to define the sizes of circle, multiply by orders of magnitude to translate to plot
    sizes = results_data[circle_size] * orders_magnitude
    '''
    if circle_size == 'ENTROPY':
        sizes = results_data['ENTROPY'] * orders_magnitude
    elif circle_size == 'STD':
        sizes = results_data['STD'] * orders_magnitude
    '''

    #sizes = sizes.to_list()
    sizes = sizes.to_numpy()

    percentile_25 = np.percentile(sizes, 25)
    percentile_50 = np.percentile(sizes, 50)
    percentile_75 = np.percentile(sizes, 75)

    #m.scatter(results_x, results_y, latlon=True, alpha=.3, s=sizes.tolist(), c='gray', cmap=colormap, zorder=1)
    #m.scatter(results_x, results_y, latlon=True, alpha=.6, s=50, c=colors, cmap=colormap, zorder=2)
    m.scatter(results_x, results_y, latlon=True, alpha=.3, s=sizes.tolist(), c='gray', zorder=1)
    m.scatter(results_x, results_y, latlon=True, alpha=.6, s=50, c=colors, zorder=2)
else:
    m.scatter(results_x, results_y, latlon=True, s=50, c=colors, cmap=colormap, zorder=1)

track = m.plot(sites_lon_x, sites_lat_y, latlon=True, label='Ship Track')

leg0 = plt.legend(handles=track, loc='lower left', bbox_to_anchor=(.13, .06))
plt.gca().add_artist(leg0)

# Create first legend
m.plot([],[], marker="o", ms=10, ls="")

texts = ["Class A", "Class B", "Class C", "Class D", "Class E"]
patches = [ m.plot([],[], marker="o", ms=10, ls="", mec=None, color=colors_for_map[i], 
            label="{:s}".format(texts[i]) )[0]  for i in range(len(texts)) ]

leg1 = plt.legend(handles=patches, loc='lower left', title = "Predictions")
plt.gca().add_artist(leg1)

if uncertainty == True:
    # create second legend
    

    '''
    min_size = min(sizes)
    max_size = max(sizes)

    total = sum(sizes)
    length = len(sizes)
    average = total/length
    '''

    scatter1 = plt.scatter([], [], c='k', alpha=0.3, s=percentile_25, label='25th Percentile' )
    scatter2 = plt.scatter([], [], c='k', alpha=0.3, s=percentile_50, label='50th Percentile')
    scatter3 = plt.scatter([], [], c='k', alpha=0.3, s=percentile_75, label='75th Percentile')

    '''
    for area in [min_size, average, max_size]:
        plt.scatter([], [], c='k', alpha=0.3, s=area,
                    label=str(area))
    '''

    leg2 = plt.legend(handles = [scatter1, scatter2, scatter3], scatterpoints=1, frameon=True, labelspacing=2.5, handletextpad=2, loc='lower right', borderpad=2) #, bbox_to_anchor=(.35, 0)
    plt.gca().add_artist(leg2)

sensor = m.scatter([sensor_longitude], [sensor_latitude], c='black', alpha=1, s=100, marker='x', linewidth=3, label='Acoustic Sensor', latlon=True, zorder=3)
leg3 = plt.legend(handles = [sensor], scatterpoints=1, frameon=True, labelspacing=1, loc='lower left', bbox_to_anchor=(.13, 0))

# add accuracy score to plot
accuracy = accuracy_score(true_predictions, results_class)
accuracy = round(accuracy, 4)
accuracy = str(accuracy)

# Use to add accuracy to plot
#plt.text(x_low - .01, y_low + ((y_high-y_low)/4),'Accuracy: ' + accuracy, fontsize=20, verticalalignment='center', horizontalalignment='left')

#l, b, w, h = plt.gca().get_position().bounds()
#plt.gca().axes.set_position(l, b-2, w, h-2)

#plt.title(plot_title)

plt.xlabel("Longitude", labelpad=30)
plt.ylabel("Latitude", labelpad=70)
plot_file = plot_file[:-4] + "_" + accuracy + plot_file[-4:]
#plot_file_eps = plot_file_eps[:-4] + "_" + accuracy + plot_file_eps[-4:]

plt.savefig(plot_file, dpi=300)
#plt.savefig(plot_file_eps, dpi=300)

plt.close()