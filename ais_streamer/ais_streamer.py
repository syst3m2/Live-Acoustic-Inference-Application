# Script to run and continuously store AIS positions in database
# Script executes on interval of choosing, conducts query for AIS track positions,
# then stores in a database

import requests
from requests.auth import HTTPBasicAuth
import json
import time
import datetime
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import geopy.distance
from bs4 import BeautifulSoup
from pyproj import Proj, transform, Transformer
from geographiclib.geodesic import Geodesic

import requests
from requests.auth import HTTPBasicAuth
import json
import time
import datetime
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import geopy.distance
from bs4 import BeautifulSoup
from pyproj import Proj, transform, Transformer
from geographiclib.geodesic import Geodesic

import sqlalchemy as db
from os import path
from time import sleep
import os


ship_class_dict ={'Landings Craft':'Class A', 'Military ops':'Class A','Fishing vessel':'Class A','Fishing Vessel':'Class A' ,'Fishing Support Vessel':'Class A', 'Tug':'Class A', 'Pusher Tug':'Class A', 'Dredging or UW ops':'Class A', 'Towing vessel':'Class A', 'Crew Boat':'Class A', 'Buoy/Lighthouse Vessel':'Class A', 'Salvage Ship':'Class A', 'Research Vessel':'Class A', 'Anti-polution':'Class A', 'Offshore Tug/Supply Ship':'Class A', 'Law enforcment':'Class A', 'Landing Craft':'Class A', 'SAR':'Class A', 'Patrol Vessel':'Class A', 'Pollution Control Vessel': 'Class A', 'Offshore Support Vessel':'Class A',
                        'Pleasure craft':'Class B', 'Yacht':'Class B', 'Sailing vessel':'Class B', 'Pilot':'Class B', 'Diving ops':'Class B', 
                        'Passenger (Cruise) Ship':'Class C', 'Passenger Ship':'Class C', 'Passenger ship':'Class C', 'Training Ship': 'Class C',
                        'Naval/Naval Auxiliary':'Class D','DDG':'Class D','LCS':'Class D','Hospital Vessel':'Class D' ,'Self Discharging Bulk Carrier':'Class D' ,'Cutter':'Class D', 'Passenger/Ro-Ro Cargo Ship':'Class D', 'Heavy Load Carrier':'Class D', 'Vessel (function unknown)':'Class D',
                        'General Cargo Ship':'Class D','Wood Chips Carrier':'Class D', 'Bulk Carrier':'Class D' ,'Cement Carrier':'Class D','Vehicles Carrier':'Class D','Cargo ship':'Class D', 'Oil Products Tanker':'Class D', 'Ro-Ro Cargo Ship':'Class D', 'USNS RAINIER':'Class D', 'Supply Tender':'Class D', 'Cargo ship':'Class D', 'LPG Tanker':'Class D', 'Crude Oil Tanker':'Class D', 'Container Ship':'Class D', 'Container ship':'Class D','Bulk Carrier':'Class D', 'Chemical/Oil Products Tanker':'Class D', 'Refrigerated Cargo Ship':'Class D', 'Tanker':'Class D', 'Car Carrier':'Class D', 'Deck Cargo Ship' :'Class D', 'Livestock Carrier': 'Class D',
                        'Bunkering Tanker':'Class D', 'Water Tanker': 'Class D', 'FSO': 'Class D', 
                        'not ship':'Class E' }

SEAVISION_API_KEY = os.environ.get('SEAVISION_API_KEY')


def get_bearing(from_lat, from_long, to_lat, to_long):
    bearing = Geodesic.WGS84.Inverse(from_lat, from_long, to_lat, to_long)['azi1'] % 360
    return bearing


def latlong_to_mercator(latitude, longitude):
    #inProj = Proj(init='epsg:3857')
    #outProj = Proj(init='epsg:4326')
    #merc_longitude, merc_latitude = transform(outProj, inProj, longitude, latitude)
    transformer = Transformer.from_crs('epsg:4326','epsg:3857', always_xy=True)
    merc_longitude, merc_latitude = transformer.transform(longitude, latitude)
    return merc_latitude, merc_longitude

#https://gist.github.com/mlgill/11334821
def rmerge(left,right,**kwargs):
    """Perform a merge using pandas with optional removal of overlapping
    column names not associated with the join. 
    
    Though I suspect this does not adhere to the spirit of pandas merge 
    command, I find it useful because re-executing IPython notebook cells 
    containing a merge command does not result in the replacement of existing
    columns if the name of the resulting DataFrame is the same as one of the
    two merged DataFrames, i.e. data = pa.merge(data,new_dataframe). I prefer
    this command over pandas df.combine_first() method because it has more
    flexible join options.
    
    The column removal is controlled by the 'replace' flag which is 
    'left' (default) or 'right' to remove overlapping columns in either the 
    left or right DataFrame. If 'replace' is set to None, the default
    pandas behavior will be used. All other parameters are the same 
    as pandas merge command.
    
    Examples
    --------
    >>> left       >>> right
       a  b   c       a  c   d 
    0  1  4   9    0  1  7  13
    1  2  5  10    1  2  8  14
    2  3  6  11    2  3  9  15
    3  4  7  12    
    
    >>> rmerge(left,right,on='a')
       a  b  c   d
    0  1  4  7  13
    1  2  5  8  14
    2  3  6  9  15
    >>> rmerge(left,right,on='a',how='left')
       a  b   c   d
    0  1  4   7  13
    1  2  5   8  14
    2  3  6   9  15
    3  4  7 NaN NaN
    >>> rmerge(left,right,on='a',how='left',replace='right')
       a  b   c   d
    0  1  4   9  13
    1  2  5  10  14
    2  3  6  11  15
    3  4  7  12 NaN
    
    >>> rmerge(left,right,on='a',how='left',replace=None)
       a  b  c_x  c_y   d
    0  1  4    9    7  13
    1  2  5   10    8  14
    2  3  6   11    9  15
    3  4  7   12  NaN NaN
    """

    # Function to flatten lists from http://rosettacode.org/wiki/Flatten_a_list#Python
    def flatten(lst):
        return sum( ([x] if not isinstance(x, list) else flatten(x) for x in lst), [] )
    
    # Set default for removing overlapping columns in "left" to be true
    myargs = {'replace':'left'}
    myargs.update(kwargs)
    
    # Remove the replace key from the argument dict to be sent to
    # pandas merge command
    kwargs = {k:v for k,v in myargs.items() if k != 'replace'}
    
    if myargs['replace'] != None:
        # Generate a list of overlapping column names not associated with the join
        skipcols = set(flatten([v for k, v in myargs.items() if k in ['on','left_on','right_on']]))
        leftcols = set(left.columns)
        rightcols = set(right.columns)
        dropcols = list((leftcols & rightcols).difference(skipcols))
        
        # Remove the overlapping column names from the appropriate DataFrame
        if myargs['replace'].lower() == 'left':
            left = left.copy().drop(dropcols,axis=1)
        elif myargs['replace'].lower() == 'right':
            right = right.copy().drop(dropcols,axis=1)
        
    df = pd.merge(left,right,**kwargs)
    
    return df

# Return pandas dataframe of all ships detected from query
# Input latitude in degree decimal, age in hours, radius in miles
def ais_boundary_query(latitude, longitude, hours, radius, engine, connection):
    URL = "http://api.seavision.volpe.dot.gov/v1/vessels"
    latitude = str(latitude)
    longitude = str(longitude)
    hours = str(hours)
    radius = str(radius)

    # Dictionary used to label the class of ship
    '''
    ship_class_dict ={'Landings Craft':'Class A', 'Military ops':'Class A','Fishing vessel':'Class A','Fishing Vessel':'Class A' ,'Fishing Support Vessel':'Class A', 'Tug':'Class A', 'Pusher Tug':'Class A', 'Dredging or UW ops':'Class A', 'Towing vessel':'Class A', 'Crew Boat':'Class A', 'Buoy/Lighthouse Vessel':'Class A', 'Salvage Ship':'Class A', 'Research Vessel':'Class A', 'Anti-polution':'Class A', 'Offshore Tug/Supply Ship':'Class A', 'Law enforcment':'Class A', 'Landing Craft':'Class A', 'SAR':'Class A', 'Patrol Vessel':'Class A', 'Pollution Control Vessel': 'Class A', 'Offshore Support Vessel':'Class A',
                        'Pleasure craft':'Class B', 'Yacht':'Class B', 'Sailing vessel':'Class B', 'Pilot':'Class B', 'Diving ops':'Class B', 
                        'Passenger (Cruise) Ship':'Class C', 'Passenger Ship':'Class C', 'Passenger ship':'Class C', 'Training Ship': 'Class C',
                        'Naval/Naval Auxiliary':'Class D','DDG':'Class D','LCS':'Class D','Hospital Vessel':'Class D' ,'Self Discharging Bulk Carrier':'Class D' ,'Cutter':'Class D', 'Passenger/Ro-Ro Cargo Ship':'Class D', 'Heavy Load Carrier':'Class D', 'Vessel (function unknown)':'Class D',
                        'General Cargo Ship':'Class D','Wood Chips Carrier':'Class D', 'Bulk Carrier':'Class D' ,'Cement Carrier':'Class D','Vehicles Carrier':'Class D','Cargo ship':'Class D', 'Oil Products Tanker':'Class D', 'Ro-Ro Cargo Ship':'Class D', 'USNS RAINIER':'Class D', 'Supply Tender':'Class D', 'Cargo ship':'Class D', 'LPG Tanker':'Class D', 'Crude Oil Tanker':'Class D', 'Container Ship':'Class D', 'Container ship':'Class D','Bulk Carrier':'Class D', 'Chemical/Oil Products Tanker':'Class D', 'Refrigerated Cargo Ship':'Class D', 'Tanker':'Class D', 'Car Carrier':'Class D', 'Deck Cargo Ship' :'Class D', 'Livestock Carrier': 'Class D',
                        'Bunkering Tanker':'Class D', 'Water Tanker': 'Class D', 'FSO': 'Class D', 
                        'not ship':'Class E' }
    '''


    HEADERS = {
        "Accept": "application/json", 
        'x-api-key': SEAVISION_API_KEY,
    }

    PARAMS = (
        ('latitude', latitude),
        ('longitude', longitude),
        ('age', hours),
        ('radius', radius),
    )

    response = requests.get(url=URL, headers=HEADERS, params=PARAMS)
    if response.text != '[]':
        if response.status_code == 200:
            response = pd.DataFrame.from_dict(json.loads(response.text))

            # Queries are performed in miles, but we desire kilometers, so we will discard returned positions not within query
            response['dist_from_sensor_km'] = response.apply(lambda x: compare_lat_long(latitude, longitude, x.latitude, x.longitude), axis=1)

            # Filter out positions that are not within defined radius from query centerpoint
            #response = response[response['dist_from_sensor_km'] <= int(float(radius))]

            # Convert time field to datetime type
            #response['timeOfFix'] = pd.to_datetime(response['timeOfFix'], unit='s', utc=True)

            # merging mmsis with response

            # Perform mmsi scrape to ensure consistent data
            mmsi_db = mmsi_scraper(response, engine)
            response = response.rename(columns={"vesselType": "desig"})

            merged_response = rmerge(response, mmsi_db, replace='left', on='mmsi', how='outer')
            merged_response.fillna(-1)


            merged_response['merc_latitude'], merged_response['merc_longitude'] = latlong_to_mercator(list(merged_response.latitude), list(merged_response.longitude))

            merged_response['ship_class'] = merged_response['desig'].apply(lambda x: assign_class(x, connection))

            #merged_response['ship_class'] = merged_response['desig'].map(ship_class_dict)
            #merged_response['ship_class'] = merged_response['ship_class'].fillna('Unknown')

            merged_response['bearing_from_sensor'] = merged_response.apply(lambda x: get_bearing(float(latitude), float(longitude), x['latitude'], x['longitude']), axis=1)

            merged_response['mmsi'] = merged_response['mmsi'].fillna(0).astype(int)
            merged_response['dead_weight'] = merged_response['dead_weight'].fillna(-1).astype(int)
            merged_response['length'] = merged_response['length'].fillna(-1).astype(int)
            merged_response['beam'] = merged_response['beam'].fillna(-1).astype(int)
            merged_response['imoNumber'] = merged_response['imoNumber'].fillna(0).astype(int)
            merged_response['heading'] = merged_response['heading'].fillna(0).astype(float)
            
            

            #merged_response = merged_response.drop('COG', axis=1)

            return merged_response
        elif response.status_code == 429:
            raise RuntimeError("Error Status Code: " +str(response.status_code) + response.text)
        else:
            print("Error Status Code: " +str(response.status_code) + response.text)
            raise Exception("Error Status Code: " +str(response.status_code) + response.text)
    else:
        response = pd.DataFrame(columns=['mmsi', 'imoNumber', 'name', 'callSign', 'cargo', 'heading', 'navStatus', 'SOG', 'latitude', 'longitude', 'timeOfFix', 'dist_from_sensor_km', 'dead_weight', 'length', 'beam', 'desig', 'merc_latitude', 'merc_longitude','ship_class','bearing_from_sensor'])
        return response
        #return False

# Return pandas dataframe of ship track history
# Input ship mmsi, age in days (default is 7, max is 90), and start time (can be up to 2 years old) as the time query should start from in the format YYYY-MM-DDTHHMMSS in UTC, default is today
def ais_track_query(mmsi, days):
    URL = "https://api.seavision.volpe.dot.gov/v1/vessels/" + str(mmsi) + "/history"
    days = str(days)
    mmsi = str(mmsi)
    start_time = str(datetime.datetime.now().isoformat())

    HEADERS = {
        "Accept": "application/json", 
        'x-api-key': 'VLpP8iMKys7MvxmAqKsRf7WV2HZ71byO51HLmIjm',
    }

    PARAMS = (
        ('age', days),
        ('time', start_time),
    )

    response = requests.get(url=URL, headers=HEADERS, params=PARAMS)
    if response.text != '[]':
        if response.status_code == 200:
            response = pd.DataFrame.from_dict(json.loads(response.text))
            response.insert(0, 'mmsi', mmsi, True)
            response['mmsi'] = response['mmsi'].fillna(0).astype(int)
            #response['timeOfFix'] = pd.to_datetime(response['timeOfFix'], unit='s')
            return response
        elif response.status_code == 429:
            raise RuntimeError("Error Status Code: " +str(response.status_code) + response.text)
        else:
            print("Error Status Code: " +str(response.status_code) + response.text)
            raise Exception("Error Status Code: " +str(response.status_code) + response.text)
    else:
        response = pd.DataFrame(columns=['mmsi', 'latitude', 'longitude', 'heading', 'SOG', 'timeOfFix'])
        return response

# First set of latitude and longitude is the center point, second set is being compared
def compare_lat_long(lat1, lon1, lat2, lon2):
    """
    Calculate the distance in kilometers between two points 
    on the earth (specified in decimal degrees, uses WGS-84 ellipsoidal distance)
    """
    #lon1, lat1, lon2, lat2 = int(lon1), int(lat1), int(lon2), int(lat2)
    coords_1 = (lat1, lon1)
    coords_2 = (lat2, lon2)
    distance = geopy.distance.geodesic(coords_1, coords_2).km # use .miles for miles
    return distance

# latitude and longitude, age looking for ships in days, radius in miles
# Return ships detected within a certain time period (hours) and within a certain range from a defined location
def ais_boundary_track_query(latitude, longitude, days, hours, radius, latest_time, engine, connection):
    URL = "https://api.seavision.volpe.dot.gov/v1/vessels"
    latitude = str(latitude)
    longitude = str(longitude)
    days = str(days)
    radius = str(radius)
    hours = str(hours)
    #current_time = datetime.datetime.utcnow()
    #query_start_time = current_time - datetime.timedelta(hours=hours)

    # Query last location of ships within radius and time period
    #ships_df = ais_boundary_query(latitude, longitude, hours, 100)
    #try:
    ships_df = ais_boundary_query(latitude, longitude, hours, 100, engine, connection)
    #except RuntimeError:
    #    print("Rate limit reached, waiting 60 minutes")
    #    sleep(3610)
    #    ships_df = ais_boundary_query(latitude, longitude, hours, 100, engine, connection)

    # Cannot get COG from historical track query, so we must drop this column
    ships_df = ships_df.drop('COG', axis=1)

    columns = ['mmsi', 'imoNumber', 'name', 'callSign', 'cargo', 'heading', 'navStatus', 
                'SOG', 'latitude', 'longitude', 'timeOfFix', 'dist_from_sensor_km', 'dead_weight', 
                'length', 'beam', 'desig', 'merc_latitude', 'merc_longitude','ship_class','bearing_from_sensor']
    ships_and_track_df = pd.DataFrame(columns=columns)

    # Query the track history for each ship from query for same time period, discard positions from outside the radius
    #query_count = 0
    if not ships_df.empty:
        for index, row in ships_df.iterrows():
            '''
            query_count += 1
            if query_count > 90:
                print("Rate limit reached, waiting 60 minutes")
                sleep(3600)
                query_count = 0
            '''
            mmsi = row['mmsi']
            
            # MMSI	latitude	longitude	heading	SOG	timeOfFix
            try:
                ship_track_df = ais_track_query(mmsi, days)
            except RuntimeError:
                print("Rate limit reached, waiting 60 minutes")
                sleep(3610)
                ship_track_df = ais_track_query(mmsi, days)
                
            
            if not ship_track_df.empty:
                ship_track_df['dist_from_sensor_km'] = ship_track_df.apply(lambda x: compare_lat_long(latitude, longitude, x.latitude, x.longitude), axis=1)
                ship_track_df['merc_latitude'], ship_track_df['merc_longitude'] = latlong_to_mercator(list(ship_track_df.latitude), list(ship_track_df.longitude))
                ship_track_df['bearing_from_sensor'] = ship_track_df.apply(lambda x: get_bearing(float(latitude), float(longitude), x['latitude'], x['longitude']), axis=1)

                # add all data from the ship boundary query to the ship track query
                '''
                ship_track_df['imoNumber'] = ships_df[ships_df['mmsi']==mmsi]['imoNumber'][0]
                ship_track_df['name'] = ships_df[ships_df['mmsi']==mmsi]['name'][0]
                ship_track_df['callSign'] = ships_df[ships_df['mmsi']==mmsi]['callSign'][0]
                ship_track_df['cargo'] = ships_df[ships_df['mmsi']==mmsi]['cargo'][0]
                ship_track_df['heading'] = ships_df[ships_df['mmsi']==mmsi]['heading'][0]
                ship_track_df['navStatus'] = ships_df[ships_df['mmsi']==mmsi]['navStatus'][0]
                ship_track_df['dead_weight'] = ships_df[ships_df['mmsi']==mmsi]['dead_weight'][0]
                ship_track_df['length'] = ships_df[ships_df['mmsi']==mmsi]['length'][0]
                ship_track_df['beam'] = ships_df[ships_df['mmsi']==mmsi]['beam'][0]
                ship_track_df['desig'] = ships_df[ships_df['mmsi']==mmsi]['desig'][0]
                ship_track_df['ship_class'] = ships_df[ships_df['mmsi']==mmsi]['ship_class'][0]
                '''
                ship_track_df['imoNumber'] = row['imoNumber']
                ship_track_df['name'] = row['name']
                ship_track_df['callSign'] = row['callSign']
                ship_track_df['cargo'] = row['cargo']
                ship_track_df['navStatus'] = row['navStatus']
                ship_track_df['dead_weight'] = row['dead_weight']
                ship_track_df['length'] = row['length']
                ship_track_df['beam'] = row['beam']
                ship_track_df['desig'] = row['desig']
                ship_track_df['ship_class'] = row['ship_class']

                # Filter out positions that are not within defined radius from query centerpoint
                ship_track_df = ship_track_df[ship_track_df['dist_from_sensor_km'] <= int(float(radius))]

                # rearrange columns to be in the correct order
                ship_track_df = ship_track_df[columns]

                # Append the row to the ship track dataframe
                ship_track_df = ship_track_df.append(row, ignore_index=True)
              
                # Append the ships track to the main dataframe
                ships_and_track_df = ships_and_track_df.append(ship_track_df, ignore_index=True)

            else:
                # If the returned dataframe is empty, then that row is the only one to add, so add it
                ships_and_track_df = ships_and_track_df.append(row, ignore_index=True)
    else:
        return ships_df, ships_and_track_df
    

    #new_ships_df = ships_df.drop(['COG','heading','navStatus','SOG','latitude','longitude','timeOfFix'], axis=1)
    #ships_and_track_df = ships_and_track_df.rename(columns={"MMSI": "mmsi"})
    #ships_and_track_df = ships_and_track_df.merge(new_ships_df, how='outer', on='mmsi')
    
    #ships_and_track_df['timeOfFix'] = pd.to_datetime(ships_and_track_df['timeOfFix'], unit='s')

    # Filter out positions older than the defined time period in hours
    ships_and_track_df = ships_and_track_df[ships_and_track_df['timeOfFix'] > latest_time]
    ships_and_track_df = ships_and_track_df.reset_index(drop=True)

    return ships_df, ships_and_track_df

# Scrape data for mmsis from vesselfinder, check if already in database, if not, then store in database
# Code adapted from LT Andrew Pfau
def mmsi_scraper(mmsi_df, engine):
    #MMSI,START_TIME,END_TIME,LABEL,DESIG,CPA,CPA_TIME
    #MMSI,START_TIME,END_TIME,LABEL,DESIG,CPA,CPA_TIME,BEARING_RANGE_DATA
    # Data recorded is MMSI, Dead Weigth Tonnage, description, and Size in meters (length / beam)
    #mmsis = pd.read_csv(os.path.join(dir, mmsiFile), header=0, names=["MMSI","IMO", "TYPE"] )
    
    # dataframe to save new ship data from web in, store in database, dataframe to save all ship data in from both web and database
    new_mmsi_db = pd.DataFrame(columns=["mmsi", "dead_weight", "length", "beam", "desig"]) 
    mmsi_db = pd.DataFrame(columns=["mmsi", "dead_weight", "length", "beam", "desig"])
    update_mmsi_db = pd.DataFrame(columns=["mmsi", "dead_weight", "length", "beam", "desig"])

    base_html = "https://www.vesselfinder.com/vessels?name="
    # needed to fool server into thinking we're a browser
    headers = headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36'}
    
    for mmsi_data in mmsi_df.iterrows():
        # check if mmsi is already in database
        # If it is, check if values are -1, update if possible

        mmsi_check = mmsi_lookup(mmsi_data[1]['mmsi'], engine)

        if not mmsi_check.empty:
            #If any of the values are -1, scrape the web again and grab them
            # Need to update database write function to insert or update at the end for this to work, otherwise there will be multiple entries for mmsi
            #if mmsi_check.iloc[0]['dead_weight']==-1 or mmsi_check.iloc[0]['length']==-1 or mmsi_check.iloc[0]['beam']==-1 or mmsi_check.iloc[0]['desig']==-1:
            #    pass
            
            mmsi_db = mmsi_db.append({'mmsi':mmsi_check.iloc[0]['mmsi'], 'dead_weight':mmsi_check.iloc[0]['dead_weight'], 'length':mmsi_check.iloc[0]['length'], 'beam':mmsi_check.iloc[0]['beam'], 'desig':mmsi_check.iloc[0]['desig']}, ignore_index=True)
            continue
            

            
        tgt = base_html + str(mmsi_data[1]['mmsi'])
        # get the html and pass into the soup parser
        html = requests.get(tgt, headers=headers)
        # verify we get the page ok
        if html.status_code == 200:
            html_dom = BeautifulSoup(html.content, features = "html.parser")
            # verfiy we did not get 'No Results'
            check = html_dom.find("section", attrs={'class',"listing"}).get_text()
            if check != "No resultsRefine your criteria and search again":
                # dead weight tonnage
                dwt = html_dom.find("td", attrs={'class':'v5 is-hidden-mobile'}).get_text()
                if not dwt.isnumeric():
                    dwt = -1
                # size in meters, length / beam
                size = html_dom.find("td", attrs={'class':'v6 is-hidden-mobile'}).get_text()
                if any(char.isdigit() for char in size):
                    size = size.split("/")
                    length = size[0].strip()
                    beam = size[1].strip()
                else:
                    length = -1
                    beam = -1
                # ship description
                #des = html_dom.find("td", attrs={'class':'v2'}).find('small').get_text()
                try:
                    des = html_dom.find("td", attrs={'class': 'v2'}).find("div", attrs={'class': 'slty'}).get_text()
                except:
                    print("issue with parser!")
                    des = 'Unknown'
                new_mmsi_db = new_mmsi_db.append({'mmsi':mmsi_data[1]['mmsi'], 'dead_weight':dwt, 'length':length,'beam': beam, 'desig':des}, ignore_index=True)
                mmsi_db = mmsi_db.append({'mmsi':mmsi_data[1]['mmsi'], 'dead_weight':dwt, 'length':length,'beam': beam, 'desig':des}, ignore_index=True)

            # try again with IMO number if we have one
            elif mmsi_data[1][1] > 0:
                tgt = base_html + str(mmsi_data[1]['imoNumber'])
                # get the html and pass into the soup parser
                html = requests.get(tgt, headers=headers)
                # verify we get the page ok
                if html.status_code == 200:
                    html_dom = BeautifulSoup(html.content, features="html.parser")
                    # verfiy we did not get 'No Results'
                    check = html_dom.find("section", attrs={'class',"listing"}).get_text()
                    if check != "No resultsRefine your criteria and search again":
                        # dead weight tonnage
                        dwt = html_dom.find("td", attrs={'class':'v5 is-hidden-mobile'}).get_text()
                        if not dwt.isnumeric():
                            dwt = -1
                        # size in meters, length / beam
                        size = html_dom.find("td", attrs={'class':'v6 is-hidden-mobile'}).get_text()
                        if any(char.isdigit() for char in size):
                            size = size.split("/")
                            length = size[0].strip()
                            beam = size[1].strip()
                        else:
                            length = -1
                            beam = -1
                        try:
                            des = html_dom.find("td", attrs={'class': 'v2'}).find("div",attrs={'class': 'slty'}).get_text()
                        except:
                            print("issue with parser!")
                            des = 'Unknown'
                        new_mmsi_db = new_mmsi_db.append({'mmsi':mmsi_data[1]['mmsi'], 'dead_weight':dwt, 'length':length,'beam': beam, 'desig':des}, ignore_index=True)
                        mmsi_db = mmsi_db.append({'mmsi':mmsi_data[1]['mmsi'], 'dead_weight':dwt, 'length':length,'beam': beam, 'desig':des}, ignore_index=True)
                    
                    # In every instance where the imo check fails, use original data
                    else: #elif int(mmsi_data[1][0]) > 0 or str(mmsi_data[1][13]).isnumeric() or int(mmsi_data[1][13]) > 0 or int(mmsi_data[1][14]) > 0 or int(mmsi_data[1][5]) > 0:

                        if mmsi_data[1]['mmsi'] > 0:
                            mmsi = mmsi_data[1]['mmsi']
                        else:
                            continue

                        if str(mmsi_data[1][13]).isnumeric() and str(mmsi_data[1][14]).isnumeric() and mmsi_data[1][13] > 0 and mmsi_data[1][14] > 0:
                            length = str(mmsi_data[1]['length'])
                            beam = str(mmsi_data[1]['beam'])
                        else:
                            length = -1
                            beam = -1

                        try:
                            if len(mmsi_data[1][5]) > 2:
                                des = mmsi_data[1]['vesselType']
                                des = des.split("-")
                                des = des[1]
                            else:
                                des = -1
                        except:
                            des = -1
                        
                        new_mmsi_db = new_mmsi_db.append({'mmsi':mmsi, 'dead_weight':-1, 'length':length,'beam': beam, 'desig':des}, ignore_index=True)
                        mmsi_db = new_mmsi_db.append({'mmsi':mmsi, 'dead_weight':-1, 'length':length,'beam': beam, 'desig':des}, ignore_index=True)
                
                else: #elif int(mmsi_data[1][0]) > 0 or str(mmsi_data[1][13]).isnumeric() or int(mmsi_data[1][13]) > 0 or int(mmsi_data[1][14]) > 0 or int(mmsi_data[1][5]) > 0:

                    if mmsi_data[1]['mmsi'] > 0:
                        mmsi = mmsi_data[1]['mmsi']
                    else:
                        continue

                    if str(mmsi_data[1][13]).isnumeric() and str(mmsi_data[1][14]).isnumeric() and mmsi_data[1][13] > 0 and mmsi_data[1][14] > 0:
                        length = str(mmsi_data[1]['length'])
                        beam = str(mmsi_data[1]['beam'])
                    else:
                        length = -1
                        beam = -1

                    try:
                        if len(mmsi_data[1][5]) > 2:
                            des = mmsi_data[1]['vesselType']
                            des = des.split("-")
                            des = des[1]
                        else:
                            des = -1
                    except:
                        des = -1
                    
                    new_mmsi_db = new_mmsi_db.append({'mmsi':mmsi, 'dead_weight':-1, 'length':length,'beam': beam, 'desig':des}, ignore_index=True)
                    mmsi_db = new_mmsi_db.append({'mmsi':mmsi, 'dead_weight':-1, 'length':length,'beam': beam, 'desig':des}, ignore_index=True)
            
            # If no data found from vesselfinder, keep data in original
            else: #elif int(mmsi_data[1][0]) > 0 or str(mmsi_data[1][13]).isnumeric() or int(mmsi_data[1][13]) > 0 or int(mmsi_data[1][14]) > 0 or int(mmsi_data[1][5]) > 0:

                if mmsi_data[1]['mmsi'] > 0:
                    mmsi = mmsi_data[1]['mmsi']
                else:
                    continue

                if str(mmsi_data[1][13]).isnumeric() and str(mmsi_data[1][14]).isnumeric() and mmsi_data[1][13] > 0 and mmsi_data[1][14] > 0:
                    length = str(mmsi_data[1]['length'])
                    beam = str(mmsi_data[1]['beam'])
                else:
                    length = -1
                    beam = -1

                try:
                    if len(mmsi_data[1][5]) > 2:
                        des = mmsi_data[1]['vesselType']
                        des = des.split("-")
                        des = des[1]
                    else:
                        des = -1
                except:
                    des = -1
                
                new_mmsi_db = new_mmsi_db.append({'mmsi':mmsi, 'dead_weight':-1, 'length':length,'beam': beam, 'desig':des}, ignore_index=True)
                mmsi_db = new_mmsi_db.append({'mmsi':mmsi, 'dead_weight':-1, 'length':length,'beam': beam, 'desig':des}, ignore_index=True)

            '''
            else:
                print("Did not find MMSI or IMO: " + str(mmsi_data[1]['mmsi']) + " " + str(mmsi_data[1]['imoNumber']))
                # still append to list so we don't search again, all values -1 as flag
                new_mmsi_db = new_mmsi_db.append({'mmsi':mmsi_data[1]['mmsi'], 'dead_weight':-1, 'length':-1,'beam':-1, 'desig':-1}, ignore_index=True)
                mmsi_db = mmsi_db.append({'mmsi':mmsi_data[1]['mmsi'], 'dead_weight':-1, 'length':-1,'beam':-1, 'desig':-1}, ignore_index=True)
            '''
        else:
            print("Failed to retrieve page " + str(tgt))

            #If couldn't get webpage, keep data from original or append with -1 values
            #if int(mmsi_data[1][0]) > 0 or int(mmsi_data[1][13]) > 0 or int(mmsi_data[1][14]) > 0 or int(mmsi_data[1][5]) > 0:

            if mmsi_data[1]['mmsi'] > 0:
                mmsi = mmsi_data[1]['mmsi']
            else:
                continue

            if str(mmsi_data[1][13]).isnumeric() and str(mmsi_data[1][14]).isnumeric() and mmsi_data[1][13] > 0 and mmsi_data[1][14] > 0:
                length = str(mmsi_data[1]['length'])
                beam = str(mmsi_data[1]['beam'])
            else:
                length = -1
                beam = -1

            try:
                if len(mmsi_data[1][5]) > 2:
                    des = mmsi_data[1]['vesselType']
                    des = des.split("-")
                    des = des[1]
                else:
                    des = -1
            except:
                des = -1
                
            new_mmsi_db = new_mmsi_db.append({'mmsi':mmsi, 'dead_weight':-1, 'length':length,'beam': beam, 'desig':des}, ignore_index=True)
            mmsi_db = mmsi_db.append({'mmsi':mmsi, 'dead_weight':-1, 'length':length,'beam': beam, 'desig':des}, ignore_index=True)

            # If no data in original, append with all -1 values
            '''
            else:
                new_mmsi_db = new_mmsi_db.append({'mmsi':mmsi_data[1]['mmsi'], 'dead_weight':-1, 'length':-1,'beam':-1, 'desig':-1}, ignore_index=True)
                mmsi_db = mmsi_db.append({'mmsi':mmsi_data[1]['mmsi'], 'dead_weight':-1, 'length':-1,'beam':-1, 'desig':-1}, ignore_index=True)
            '''
    # all done, write out new mmsis to mmsi_db and return all mmsi information from the database
    # mmsi_db.to_csv(os.path.join(cwd,mmsiDB), index=False)
    # new_mmsi_db['ship_class'] = new_mmsi_db['desig'].map(ship_class_dict)
    # Drop duplicate MMSIs
    new_mmsi_db = new_mmsi_db.drop_duplicates(subset='mmsi', keep="last")

    new_mmsi_db = new_mmsi_db[new_mmsi_db['mmsi'].notna()]
    new_mmsi_db['mmsi'] = new_mmsi_db['mmsi'].astype(int)
    new_mmsi_db['dead_weight'] = new_mmsi_db['dead_weight'].fillna(-1).astype(float)
    new_mmsi_db['length'] = new_mmsi_db['length'].fillna(-1).astype(int)
    new_mmsi_db['beam'] = new_mmsi_db['beam'].fillna(-1).astype(int)
    
    mmsi_db_store(new_mmsi_db, engine)

    mmsi_db['mmsi'] = mmsi_db['mmsi'].astype(int)
    mmsi_db['dead_weight'] = mmsi_db['dead_weight'].fillna(-1).astype(float)
    mmsi_db['length'] = mmsi_db['length'].fillna(-1).astype(int)
    mmsi_db['beam'] = mmsi_db['beam'].fillna(-1).astype(int)


    return mmsi_db

def assign_class(desig, connection):
    class_query="SELECT ship_class FROM ship_classes WHERE desig = '"+desig+"'"
    ship_class = connection.execute(class_query)
    #if nothing returned, then add desig to database with Unknown ship class
    ship_class = ship_class.first()
    if ship_class==None:
        print("New ship designation " + desig + " added to the ship classes table")
        add_desig_df = pd.DataFrame(columns=["desig","ship_class"], data=[[desig,"Unknown"]])
        add_desig_df.to_sql(name='ship_classes', con=connection, schema='public', index=False, if_exists='append')
        ship_class = "Unknown"
    else:
        ship_class = ship_class[0]
        
    return ship_class

# Returns new positions from new_ais_df that aren't in old_ais_df
def ais_compare(old_ais_df, new_ais_df):
    comparison_df = new_ais_df.merge(old_ais_df, indicator=True, how='outer')
    new_ship_pos = comparison_df[comparison_df['_merge'] == 'left_only']

    if not new_ship_pos.empty:
        new_ship_pos = new_ship_pos.drop(["_merge"], axis=1)

    return new_ship_pos

# Database read/write functions
# Store AIS positions in database from stream

def ais_db_write(new_ais_positions, engine):
    #https://hackersandslackers.com/compare-rows-pandas-dataframes/
    #user = User(username=form.username.data, email=form.email.data)
    #db.session.add(user)
    #db.session.commit()
    new_ais_positions['record_timestamp'] = datetime.datetime.utcnow() #(datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds()
    new_ais_positions.to_sql(name='ais', con=engine, schema='public', index=False, if_exists='append')



def mmsi_db_store(mmsi_db, engine):
    #https://hackersandslackers.com/compare-rows-pandas-dataframes/
    #If mmsi is already in db, then update with new values, if it isn't, create new entry

    mmsi_db['record_timestamp'] = datetime.datetime.utcnow()
    mmsi_db.to_sql(
        'mmsi',
        con=engine,
        schema='public',
        index=False,
        if_exists='append'
    )

    return mmsi_db

def mmsi_lookup(mmsi, engine):

    mmsi_check = pd.read_sql_query(
        'SELECT * FROM mmsi WHERE mmsi=' + str(mmsi),
        con=engine
    )

    '''    
    if mmsi == 367005910:
        d = {'mmsi': [367005910], 'dead_weight': [2345], 'length': [21], 'beam': [21], 'desig': ['Passenger ship']}
        mmsi_check = pd.DataFrame(data=d)
    else:
        mmsi_check = pd.DataFrame()
    mmsi_check = pd.DataFrame()
    '''

    return mmsi_check