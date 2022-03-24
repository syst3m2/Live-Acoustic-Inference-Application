#!/bin/bash
while ! nc -z app-db 5432; do
    sleep 0.1
done

python main_ais_stream.py $1