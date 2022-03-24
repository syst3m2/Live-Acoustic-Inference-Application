#!/bin/bash
echo "initializing postgresql database"
rm -r migrations
flask db init
flask db migrate -m "initialize database"
flask db upgrade
echo "database initialized"