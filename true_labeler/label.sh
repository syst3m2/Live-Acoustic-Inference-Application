#!/bin/bash
while ! nc -z app-db 5432; do
    sleep 0.1
done
#python true_labeler.py $1
python true_labeler.py