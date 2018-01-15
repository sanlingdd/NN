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

def importDataFromCSVToDB(workQueue):
    
    conn = pymysql.Connect(host="localhost",
                           port=3306,
                           user="root",
                           password="Initial0",
                           database="stock",
                           charset="utf8")
    cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
    

    while(not workQueue.empty()):
        fileName = workQueue.get()                    
        df = pd.read_csv(location + "\\" + fileName, header=None,
                     names=['Date', 'Time', 'OpenPrice', 'LowPrice', 'HighPrice', 'ClosePrice', 'Volume', 'Amount'],
                     dtype={'Date':str, 'Time':str, 'LowPrice':np.float64, 'HighPrice':np.float64, 'ClosePrice':np.float64, 'Volume':np.float64, 'Amount':np.float64
                            }
                     )
        df.fillna(value=0)
        for index, row in df.iterrows():
            try:
                cursor.execute('insert into minute(Date,Time,OpenPrice,LowPrice,HighPrice,ClosePrice,Volume,Amount,code) values(%s,%s,%s,%s,%s,%s,%s,%s,%s)', [row['Date'], row['Time'], row['OpenPrice'], row['LowPrice'], row['HighPrice'], row['ClosePrice'], row['Volume'], row['Amount'], fileName[2:8]])
                conn.commit()
            except  Exception:
                print(fileName)
                print(index, row)
        
# Insert to a dataframe
    # df = pd.DataFrame(data=list(result))
    cursor.close()
    conn.close()


workQueue = Queue(5000)
if __name__ == '__main__':
    lock = threading.Lock()
    location = r'C:\Users\i071944\Documents\Stk_1F_2006\Stk_1F_2006'
    FileNames = []
    os.chdir(location)
    
    conn = pymysql.Connect(host="localhost",
                           port=3306,
                           user="root",
                           password="Initial0",
                           database="stock",
                           charset="utf8")
    cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
    for files in os.listdir("."):
        if files.endswith(".csv"):
            cursor.execute('select count(*) as count from minute where code = %s', [files[2:8]])
            result = cursor.fetchall()
            count = result[0].get('count')
            if count != 0:
                continue
            else:
                workQueue.put(files)
    
    cursor.close()
    conn.close()
    
    for i in range(100):
        t = threading.Thread(target=importDataFromCSVToDB, args=(workQueue,))
        t.start()
        time.sleep(random.randint(low=0, high=10))
