import sys
sys.path.append('..')

import pandas as pd
import pymysql
from pymongo import MongoClient
import json
import tushare as ts
from tushare.stock import cons as cs
from io import StringIO
import urllib
from datetime import datetime
import traceback  
import os


codes = ['002343']

for code in codes:
    df = ts.get_hist_data(code,ktype='5')
    filename = 'data/{}'.format(code);
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=None)
    else:
        df.to_csv(filename)