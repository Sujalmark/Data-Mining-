COSC-3026 Final Project - Apriori vs FP-Growth (+ TID-list Apriori, density selector)

FOLDER CONTENTS
  mining.py            all algorithms + benchmark
  download_data.py     downloads chess, mushroom, retail from fimi.uantwerpen.be
  requirements.txt     libraries (numpy, pandas, matplotlib)
  RUN_ALL_WINDOWS.bat  double-click to do everything (Windows)
  run_all_mac_linux.sh same for Mac/Linux:  sh run_all_mac_linux.sh
  data/                datasets go here (put groceries.dat here yourself)
  results/             output (CSV files + charts) appears here
  COSC3026_Final_Report.docx   report with yellow placeholders to fill

STEPS
  1. Install Python 3 (python.org). On Windows tick "Add Python to PATH".
  2. Put groceries.dat in the data folder (9,835 baskets). If you cannot find it,
     the program simply skips it.
  3. Windows: double-click RUN_ALL_WINDOWS.bat
     Mac/Linux: open a terminal in this folder and run:  sh run_all_mac_linux.sh
     Or type the commands yourself:
        pip install -r requirements.txt
        python download_data.py
        python mining.py verify
        python mining.py bench data results
  4. When it finishes, send the contents of these files from the results folder
     to Claude: dataset_stats.csv, results.csv, rules_denominator.csv, selector.txt
     (plus your CPU, RAM, OS and Python version). Insert runtime.png and memory.png
     into the Word report yourself.

IF IT IS TOO SLOW
  Open mining.py, find DEFAULT_SUPPORTS, and delete the LOWEST number for the slow dataset.

Retail data was donated by Tom Brijs (FIMI repository) - acknowledge this in the report.
