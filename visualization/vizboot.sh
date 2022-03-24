#!/bin/bash

echo "Waiting for postgres..."
while ! nc -z app-db 5432; do
    sleep 0.1
done

echo "PostgreSQL started"
bokeh serve acoustic_plot.py metrics_plot.py --port 5011 --allow-websocket-origin '*' --session-token-expiration 3600000