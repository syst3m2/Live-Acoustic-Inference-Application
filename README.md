## A Docker-based, Flask Python web-application with an embedded Bokeh application to run multi-tenant, live-inference machine learning models against an acoustic data stream. 
The models deployed onto the system predict ship classes on a live acoustic stream with an AIS visualization for correlation and fusion of multiple sources of sensor data.

This code was created in support of the master's thesis located at https://calhoun.nps.edu/handle/10945/68759 


## Home Page:
The home page provides access to the various modules present in the web application.
![Home Page](/images/app_home_page.png "Home Page")

## Visualization Module:
The visualization module provides access to the spectrogram, model uncertainties, AIS picture, and predicted ship classes.
![Visualization Module](/images/user_interface_live_inference.png "Visualization Module")

## Metrics Module:
The metrics module provides access to model performance over time.
![Metrics Module](/images/user_interface_metrics.png "Metrics Module")

## Data Management Module:
The data management module provides functionality to edit and update ship classes to deal with concept drift in the dataset. 
![Data Management Module](/images/user_interface_manage_data.png "Data Management Module")

## Model Management Module:
The model management module allows users to deploy, deactivate, and reactive machine learning models to do live-inference.
![Model Management Module](/images/user_interface_manage_models.png "Model Management Module")



## Getting Started:
Note: These instructions and the application will not function for any user, because it requires access to data sources that are not publicly accessible. Please use the below as a general description of the docker container purposes and methods used for the system architecture.


To run this application, you will require a SeaVision API key. 
Visit this site, create an account, and request an API key.
User Account: https://seavision.volpe.dot.gov/login
API Key Instructions: https://info.seavision.volpe.dot.gov/apikeys

Mount the acoustic data stream drive over SSH
https://www.arubacloud.com/tutorial/how-to-mount-remote-directories-with-sshfs-on-ubuntu-18-04.aspx

sudo apt-get install sshfs

sudo mkdir /mnt/folder_name

sudo sshfs -o allow_other,default_permissions user@server_ip:/path/to/folder /mnt/folder_name

Make the connection persistent by doing the below

sudo nano /etc/fstab

scroll to the bottom of the file and add

sshfs#user@server_ip:/path/to/mount /mnt/server_folder

You can setup ssh certificate authentication too if you desire with this command (replace id_rsa with whichever name you saved the key for)

sudo sshfs -o allow_other,default_permissions,IdentityFile=~/.ssh/id_rsa user@server_ip:/ /mnt/folder_name

and by following these instructions
https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys-2

This helps re-establish the ssh connection if it ever becomes unmounted/disconnected without need to 
re-enter your password

Now clone this git repository to the machine that will run the application

Update the docker-compose environment variables as follows

volumes:
    acoustic data - point this to acoustic data stream that you just mounted
    models - Point this to folder that you want models stored in

    Update each volume to point to filepath for code of each module. This allows you to 
    run a git pull on the production environment, restart all of the services, and they will run
    with updated code

    Here is a description of each volume you will need to mount:
    Data
    JSON tracker
    Code (optional)
        Mounting the volume with the code allows you to make updates on the fly. 

Now build the docker containers. What I like to do is have one folder with a running instance of the application for development and also one for production. Just navigate to the appropriate folder and replace the filename with the correct one.

acoustic-app
    |
    |---->dev_acoustic_app: run docker-compose -f docker-compose.dev.yml build
    |
    |---->prod_acoustic_app: run run docker-compose -f docker-compose.dev.yml build

After the docker containers have been built, you are ready to start running the application

first start the app service in dev-first mode and app-db to initialize the database.
Go into the docker-compose under the app service and specify dev-first in the command line
Now run:
docker-compose -f docker-compose.dev/prod.yml up -d app-db app

If you have a backed up database file to initialize the database from, then follow the instructions below
for restoring a backed up database.

Once the database has been initialized, you can start streaming AIS positions into it

Start the ais streamer and database first for an hour to ensure appropriate true labels
sudo docker-compose -f docker-compose.dev.yml up -d app-db ais-stream

Then run
sudo docker-compose -f docker-compose.dev.yml up -d
sudo docker-compose -f docker-compose.dev.yml up -d <(optional: container-name)>

to start the development docker image in detached mode

Then to follow along with the logs

sudo docker-compose -f docker-compose.dev.yml logs --tail=30 -t -f

to stop the containers run
sudo docker-compose -f docker-compose.dev.yml stop
to specify a specific service, specify the name after stop (optional)

to attach to a running container, run
sudo docker attach <container_name>

to view running containers
sudo docker ps

Once the application is running, create a user account. An account is required to achieve full functionality with the application to deploy/manage models



#Individual Services Description
## For more detailed information, please refer to the Wiki documentation

**app-db**
This service is a postgresql database container. Some useful maintenance functions are below

First attach to the running container instance

Run to get permissions
```
sudo -u <username> -i
sudo -u postgres -i
```

to attach to the database when in the container run. Username is automatically authenticated when request is coming from same container
```
psql <database_name> <username>
```
or
```
psql inference_db postgres
```

To view all tables
```
\dt
```
To view a table's schema
```
\d+ <table_name>
```

To backup the database, run this on the host machine
```
docker exec -t dev_acoustic_app_app-db_1 pg_dumpall -c -U postgres > dump_`date +%d-%m-%Y"_"%H_%M_%S`.sql
```

To restore a backed up database, run this on the host machine
```
cat your_dump.sql | docker exec -i dev_acoustic_app_app-db_1 psql -U postgres
```

I set up a cron job to backup the database every day at midnight with the following entry

```
crontab -e
```

Add this line to the document
```
0 0 * * * /root/backup.sh
```

the backup.sh file should be a bash script with the database backup command above

To keep space in the file, I let this script run for two weeks, then I set another cron job to execute the following bash script weekly. This deletes the 7 oldest files in the directory with
the database backups

```
rm $(ls dump* -1t | tail -7)
```

The crontab entry for this is
```
0 0 * * 6 /root/delete.sh
```
which runs every saturday at midnight

These are my scripts for backing up the database on regular intervals

```
db_backup.sh

#!/bin/bash

docker exec -t dev_acoustic_app_app-db_1 pg_dumpall -c -U postgres > /home/nicholas.villemez@ern.nps.edu/inference_application/db_backup/dev_db_backup/dump_`date +%m-%d-%Y"_"%H_%M_%S`.sql

echo "Database backed up `date +%m-%d-%Y"_"%H_%M_%S`" >> /home/nicholas.villemez@ern.nps.edu/inference_application/db_backup/dev_db_backup/backup.log
```

```
remove.sh

#!/bin/bash

echo "Deleted files on `date +%m-%d-%Y"_"%H_%M_%S`" >> empty.log
ls dump* -1t | tail -7 >> empty.log
echo "---------------------" >> empty.log
rm $(ls dump* -1t | tail -7)
```

To manage the database more easily with a GUI, you can download DBeaver and create a connection to the database. This is helpful to export data for further analysis and easy access to the database.

https://dbeaver.io/

Connection setup
https://github.com/dbeaver/dbeaver/wiki/Create-Connection

Also, the database can become overloaded with connections, so I ran the following to make sure that any connections greater than a day old are terminated.

```
alter system set idle_in_transaction_session_timeout='1d';
```
You can check the timeout settings with
```
show idle_in_transaction_session_timeout;
```
In order for the settings to take effect, you need to restart the database container

**app**
This service runs a Flask application. Any changes to the database schema should be conducted from this service as the migrations are managed by alembic. To change the database, update Models.py with your changes and run the following:

```
flask db migrate -m "comment"
flask db upgrade
```

The first command creates the alembic migration script and the second actually implements the changes to the database. I would recommend backing up the database prior to making any changes in case the changes are not implemented correctly.

The web pages can be updated in the app/templates folder

**visualization**
This service runs a Bokeh application. The Flask application sends requests to the bokeh application to generate the visualizations and embed them in the web application. To create a new visualization, you can create a new file "<vizname.py>", and then add that filename to vizboot.sh. 

This is also the container I use to create visualizations and export them in png formats. These scripts are saved in the visualization/export_visualizations folder

**predict-stream**
This service pulls the model filenames from the database, loads them, and performs predictions at 30 minutes intervals on data. A prediction is made for every 30 seconds of data. As new models are created, this code may require editing to perform correctly.

**ais-stream**
This service queries an API endpoint for ship AIS positions and stores those positions in a database. 

**mbari**
To log onto the mbari database that indexes all of the audio data, install the mysql command line utility

```
sudo apt install mysql-client-core-8.0
```

Then you can connect to the database with the following

```
mysql -u <username> -p -h <dest-url> -P 3306 -D mbari
```