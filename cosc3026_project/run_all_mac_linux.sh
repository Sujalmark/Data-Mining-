#!/bin/sh
python3 -m pip install -r requirements.txt
python3 download_data.py
python3 mining.py verify
python3 mining.py bench data results
echo "Done. Open the results folder."
