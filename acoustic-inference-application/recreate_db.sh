#!/bin/bash
echo "Recreating postgresql database"
rm -r migrations
python delete_db.py
flask db init
flask db migrate -m "initialize database"
flask db upgrade
python db.py
echo "database initialized"