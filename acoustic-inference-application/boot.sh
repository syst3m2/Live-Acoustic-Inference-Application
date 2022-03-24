#!/bin/bash

#flask db upgrade
#exec gunicorn -b 0.0.0.0:$PORT --access-logfile - --error-logfile - acoustic-inference-app:app

# Do this when you need to initialize the database
if [ "$1" = "dev-first" ]
then
    echo "Docker environment setup"
    echo "Waiting for postgres..."

    while ! nc -z app-db 5432; do
        sleep 0.1
    done

    echo "PostgreSQL started"
    ./init_dev_db.sh
    exec flask run --host=0.0.0.0

# Do this when database is already initialized
elif [ "$1" = "dev-cont" ]
then
    echo "Docker environment setup"
    echo "Waiting for postgres..."

    while ! nc -z app-db 5432; do
        sleep 0.1
    done

    echo "PostgreSQL started"
    exec flask run --host=0.0.0.0

# Do this to recreate the database
elif [ "$1" = "dev-recreate" ]
then
    echo "Docker environment setup"
    echo "Waiting for postgres..."

    while ! nc -z app-db 5432; do
        sleep 0.1
    done

    echo "PostgreSQL started"
    ./recreate_db.sh
    exec flask run --host=0.0.0.0

# Do this when running production first
elif [ "$1" = "prod-first" ]
then
    echo "Docker environment setup"
    echo "Waiting for postgres..."

    while ! nc -z app-db 5432; do
        sleep 0.1
    done

    echo "PostgreSQL started"

    ./init_prod_db.sh
    #exec flask run --host=0.0.0.0
    exec gunicorn -b 0.0.0.0:$PORT --access-logfile - --error-logfile - acoustic-inference-app:app

# Do this when running production
elif [ "$1" = "prod" ]
then
    echo "Docker environment setup"
    echo "Waiting for postgres..."

    while ! nc -z app-db 5432; do
        sleep 0.1
    done

    echo "PostgreSQL started"

    #./init_prod_db.sh
    #exec flask run --host=0.0.0.0
    exec gunicorn -b 0.0.0.0:$PORT --access-logfile - --error-logfile - acoustic-inference-app:app

# If no docker, then we set the environment variables here
elif [ "$1" = "no-docker" ]
then
    echo "linux environment setup"
    export FLASK_APP=/home/lemgog/thesis/acoustic_app/acoustic-inference-application/acoustic-inference-app.py
    export FLASK_ENV=development
    export APP_SETTINGS=config.DevelopmentConfig
    export DATABASE_URL=sqlite:////home/lemgog/thesis/acoustic_app/acoustic-inference-application/app.db
    export ACOUSTIC_DATABASE_URL=sqlite:////home/lemgog/thesis/acoustic_app/data/mbari/master_index.db
    export BOKEH_PY_LOG_LEVEL=debug
    #bokeh serve visualization/acoustic_plot.py --port 5011 --allow-websocket-origin '*'
    exec flask run --port 5001
else
    "Please select enter either docker or linux for environment"
fi