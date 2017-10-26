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
cursor.execute(sql, ('6000000',))
table_rows = cursor.fetchone()
#df = pd.read_sql('SELECT * FROM DAY where CODE = \'6000000\'',con=conn)
df = pd.DataFrame(table_rows)

df