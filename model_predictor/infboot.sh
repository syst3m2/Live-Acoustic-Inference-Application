#!/bin/bash
while ! nc -z app-db 5432; do
    sleep 0.1
done
#fresh-db ../data
#update-db ../data
#python main_predict_stream.py $1
while true
do
    python main_predict_stream.py
    echo "Predict program segfaulted, retrying"
done