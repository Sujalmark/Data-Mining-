@echo off
echo === Installing libraries ===
python -m pip install -r requirements.txt
echo === Downloading FIMI datasets ===
python download_data.py
echo === Checking the algorithms ===
python mining.py verify
echo === Running benchmark (this can take a long time) ===
python mining.py bench data results
echo === Done. Open the results folder ===
pause
