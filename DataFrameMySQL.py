# coding=utf-8

import pandas as pd
import sys
import numpy as np
import pymysql
import os
import threading
from numpy import random
import time
from multiprocessing import Queue
import urllib
import zlib
import json
import requests
import traceback


conn = pymysql.Connect(host="localhost",
                       port=3306,
                       user="root",
                       password="Initial0",
                       database="stock",
                       charset="utf8")
cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)

sql = "SELECT * FROM `day` where `code`=%s" 
cursor.execute(sql, ('600000',))
table_rows = cursor.fetchone()

df = pd.read_sql('SELECT Date, OpenPrice,ClosePrice,Diff,Percent,LowPrice, HighPrice,Volume, Amount,Exchange FROM DAY where CODE = \'600000\'',
                 con=conn,columns=['Date', 'OpenPrice','ClosePrice','Diff','Percent', 'LowPrice', 'HighPrice',  'Volume', 'Amount','Exchange'],
                 index_col='Date')

def getTrainingExample(date,df):
    
    for index, row in df.iterrows():
        if index == date:
            
    #
    return





with conn.cursor() as cursor:
    # Read a single record
    sql = "SELECT `ID` FROM `day` WHERE `code`=%s" 

    cursor.execute(sql, ('6000000',))
    result = cursor.fetchone()
    print(result)

