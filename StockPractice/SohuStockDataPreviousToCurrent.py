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
from datetime import datetime, date

def tryconvert(value):
    try:
        return np.float64(value)
    except ValueError:
        return 0
    return 0

def importDataFromCSVToDB(workQueue):
    
    conn = pymysql.Connect(host="localhost",
                           port=3306,
                           user="root",
                           password="Initial0",
                           database="stock",
                           charset="utf8")
    cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
    

    while(not workQueue.empty()):
        code = workQueue.get()
        codeStr = str(code)   
        codeStr = codeStr[0:codeStr.find('.')]
        codeStr = codeStr.zfill(6)                 
        site = 'http://q.stock.sohu.com/hisHq?code=cn_{}&start={}&end={}&stat=1&order=D&period=d&rt=json';
        #current 20171020
        #From 20171021 To 20171117
        site = site.format(codeStr,'20171021','20171117')
        response =requests.get(site)
        
        html = response.text
        htmljson = json.loads(html)
        try:
            data = htmljson[0].get('hq')
        except Exception:
            print(code)
            continue
        
        df = pd.DataFrame(data=data, columns=['Date', 'OpenPrice','ClosePrice','Diff','Percent', 'LowPrice', 'HighPrice',  'Volume', 'Amount','Exchange']
                         )
        df['Percent'] = df['Percent'].apply(lambda x : x[:len(x) - 1])
        df['Exchange'] = df['Exchange'].apply(lambda x : x[:len(x) - 1])
        df['Percent'] = df['Percent'].apply(lambda x :  tryconvert(x))
        df['Exchange'] = df['Exchange'].apply(lambda x : tryconvert(x))
        
        df.fillna(value=0)
        
        try:
            df.astype (dtype={'Date':str,'OpenPrice':np.float64,'ClosePrice':np.float64,'Diff':np.float64,'Percent':np.float64,'LowPrice':np.float64,'HighPrice':np.float64,  'Volume':np.float64, 'Amount':np.float64,'Exchange':np.float64})
        except ValueError:
            print(sys.exc_info())
            
        for index, row in df.iterrows():
            try:
                cursor.execute('insert into newday(DateString,Date,OpenPrice,ClosePrice,Diff,Percent,LowPrice,HighPrice,Volume,Amount,Exchange,code) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)', [row['Date'],datetime.strptime(row['Date'],'%Y-%m-%d').timestamp(),row['OpenPrice'], row['ClosePrice'], row['Diff'], row['Percent'], row['LowPrice'], row['HighPrice'], row['Volume'], row['Amount'],row['Exchange'], code])
                conn.commit()
            except Exception:
                print(sys.exc_info())
                print(traceback.print_exc())
            
        
# Insert to a dataframe
    # df = pd.DataFrame(data=list(result))
    cursor.close()
    conn.close()


workQueue = Queue(5000)
if __name__ == '__main__':
    lock = threading.Lock() 
    
    conn = pymysql.Connect(host="localhost",
                           port=3306,
                           user="root",
                           password="Initial0",
                           database="stock",
                           charset="utf8")
    cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
    
    cursor.execute('SELECT code FROM stock.newday group by code')
    result = cursor.fetchall()
    df = pd.DataFrame(result)

    
    cursor.close()
    conn.close()
        
    codes = list(df['code'])
    for code in codes:
        workQueue.put(code)
        
    for i in range(1):
        t = threading.Thread(target=importDataFromCSVToDB, args=(workQueue,))
        t.start()
        time.sleep(random.randint(low=0, high=10))
