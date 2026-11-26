"""Download the three FIMI datasets used in the project into ./data
Run:  python download_data.py
Source: https://fimi.uantwerpen.be/data/  (Retail was donated by Tom Brijs; please acknowledge it)
"""
import os, urllib.request

BASE = "https://fimi.uantwerpen.be/data/"
FILES = ["chess.dat", "mushroom.dat", "retail.dat"]
os.makedirs("data", exist_ok=True)
for f in FILES:
    dest = os.path.join("data", f)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("already have", f)
        continue
    print("downloading", f, "...")
    urllib.request.urlretrieve(BASE + f, dest)
    n = sum(1 for _ in open(dest, encoding="utf-8", errors="ignore"))
    print("  saved", dest, "-", n, "lines (transactions)")
print("Expected lines: chess 3196, mushroom 8124, retail 88162")
