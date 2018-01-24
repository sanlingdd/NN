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


codes = ['002466','600606','600848','000016']

for code in codes:
    df = ts.get_hist_data(code,start='2015-01-01',end='2017-12-31')
    filename = 'data/{}'.format(code);
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=None)
    else:
        df.to_csv(filename)