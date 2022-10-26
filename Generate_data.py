import os
#import urllib   [was generating error]
import urllib; from urllib.request import urlopen

base_url = "https://resources.lendingclub.com/"
url_files = [
    "LoanStats_2019Q1.csv.zip",
    "LoanStats_2019Q2.csv.zip",
    "LoanStats_2019Q3.csv.zip",
    "LoanStats_2019Q4.csv.zip",
    "LoanStats_2020Q1.csv.zip",
]
for url_file in url_files:
    if not os.path.isfile(url_file):
        print("downloading", url_file)
        urllib.request.urlretrieve(base_url+url_file, url_file)
        print("\tdone")
    else:
        print("already has", url_file)