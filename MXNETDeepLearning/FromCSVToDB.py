# coding=utf-8

import pandas as pd
import sys
import numpy as np
import pymysql
import os
import threading

lock = threading.Lock()

lock.acquire()
lock.release()
if(lock.locked()):
    lock.release()

location = r'C:\Users\i071944\Documents\Stk_1F_2006\Stk_1F_2006'

conn = pymysql.Connect(host="localhost",
                       port=3306,
                       user="root",
                       password="Initial0",
                       database="stock",
                       charset="utf8")
cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)


FileNames = []
os.chdir(location)
for files in os.listdir("."):
    if files.endswith(".csv"):
        FileNames.append(files)


for fileName in FileNames:
    try:
        df = pd.read_csv(location + "\\" + fileName,header=None,
                     names=['Date','Time','OpenPrice','LowPrice','HighPrice','ClosePrice','Volume','Amount'],
                     dtype={'Date':str,'Time':str,'LowPrice':np.float64,'HighPrice':np.float64,'ClosePrice':np.float64,'Volume':np.float64,'Amount':np.float64
                            }
                     )
        df.fillna(value = 0)
        for index, row in df.iterrows():
            try:
                cursor.execute('insert into minute(Date,Time,OpenPrice,LowPrice,HighPrice,ClosePrice,Volume,Amount,code) values(%s,%s,%s,%s,%s,%s,%s,%s,%s)',[row['Date'],row['Time'],row['OpenPrice'],row['LowPrice'],row['HighPrice'],row['ClosePrice'],row['Volume'],row['Amount'],fileName[2:8]])
                conn.commit()
            except  Exception:
                print(fileName)
                print(index, row)
    except Exception:
        print(fileName)
        
# Insert to a dataframe
#df = pd.DataFrame(data=list(result))
cursor.close()
conn.close()


