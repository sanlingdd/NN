#encoding=UTF-8
from pymongo import MongoClient
import sys
sys.path.append('..')
import pandas as pd
import json
import threading
import time
from datetime import date, datetime 
from multiprocessing import Queue
from numpy import random
import pymysql
import tushare as ts
from tushare.stock import cons as cs
from io import StringIO
import urllib
import math
import traceback



def get_tick_data_history_sina(date,code,proxy,port,pause):
    symbol = cs._code_to_symbol(code)
    url = 'http://market.finance.sina.com.cn/downxls.php?date={}&symbol={}'.format(date,symbol)
    proxy_handler = urllib.request.ProxyHandler({'http': 'http://{}:{}/'.format(proxy,port)})
    opener = urllib.request.build_opener(proxy_handler)
    lines = opener.open(url,timeout=pause).read()
    lines = lines.decode('GBK') 
    if len(lines) < 20:
        return None
    df = pd.read_table(StringIO(lines), names=cs.TICK_COLUMNS,
                       skiprows=[0]) 
    return df

def getTickDataForStockIntoMongo(daysQueue,proxyQueue,db):
    while(True):
        time.sleep(1)
        while(not daysQueue.empty()):
            date,code = daysQueue.get()        
            proxy,port = proxyQueue.get()
            try:
                df = get_tick_data_history_sina(date,code,proxy,port,pause=30)
                proxyQueue.put((proxy,port))
                if len(df) < 10:
                    continue
                
                df['time'] = df['time'].apply(lambda x: datetime.strptime(date +' ' + x,'%Y-%m-%d %H:%M:%S').timestamp())
                df['code'] = code 
                db.Tick600606Data.insert(json.loads(df.to_json(orient='records')))
                time.sleep(1)                
            except:
                traceback.print_exc()
                daysQueue.put((date,code))
                proxyQueue.put((proxy,port))
                continue  

def dayMaintainThread(StockworkQueue,daysQueue):
    conn = pymysql.Connect(host="localhost",
                           port=3306,
                           user="root",
                           password="Initial0",
                           database="stock",
                           charset="utf8")
    cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
    previousStockCode=None    
    while not StockworkQueue.empty():  
        
        if daysQueue.empty(): 
            if previousStockCode != None:
                db.StockCodeFinished.insertOne({ 'CodeString': previousStockCode})
            code = StockworkQueue.get()
            previousStockCode = code
            cursor.execute('SELECT DateString FROM stock.day where code={}'.format(code))
            result = cursor.fetchall()
            allDays = []
            if len(result) != 0:
                result = pd.DataFrame(result)
                allDays = set(result['DateString'])
            for day in allDays:
                daysQueue.put((day,code))
    
        time.sleep(random.randint(low=0, high=1))
    cursor.close()
    conn.close()

proxyQueue = Queue(20)
StockworkQueue = Queue(5000)
daysQueue = Queue(10000)
if __name__ == '__main__':
    lock = threading.Lock() 
    conn = pymysql.Connect(host="localhost",
                           port=3306,
                           user="root",
                           password="Initial0",
                           database="stock",
                           charset="utf8")
    cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
    
    cursor.execute('SELECT code FROM stock.day group by code')
    result = cursor.fetchall()
    allCode = []
    if len(result) != 0:
        result = pd.DataFrame(result)
        allCode = set(result['code'])
        

    client = MongoClient('localhost', 27017)
    db = client.stock
    cursor =db.StockCodeFinished.find().sort([('CodeString',-1)])
    # Expand the cursor and construct the DataFrame
    codesdf =  pd.DataFrame(list(cursor))    
    existedCodes =[]
    if len(codesdf) != 0:
        existedCodes = set(codesdf['CodeString'])
    i = 0
#     for code in allCode:
#         if code not in existedCodes:
#             code = str(math.floor(code)).zfill(6)
#             StockworkQueue.put(code)
    
    StockworkQueue.put('600606')
    proxiesList = [('proxy.pvgl.sap.corp',8080),('proxy.sha.sap.corp',8080),('proxy.pek.sap.corp',8080),('proxy.hkg.sap.corp',8080),('proxy.sin.sap.corp',8080),('proxy.syd.sap.corp',8080),('proxy.tyo.sap.corp',8080),('proxy.wdf.sap.corp',8080),('proxy.pal.sap.corp',8080),('proxy.phl.sap.corp',8080), ('proxy.osa.sap.corp',8080)] 
    for proxy in proxiesList:
        proxyQueue.put(proxy)
        
        
    t = threading.Thread(target=dayMaintainThread, args=(StockworkQueue,daysQueue))
    t.start()
    time.sleep(random.randint(low=0, high=1))
    
    for i in range(10):
            tday = threading.Thread(target=getTickDataForStockIntoMongo, args=(daysQueue,proxyQueue,db))
            tday.start()
            time.sleep(random.randint(low=0, high=1))



##datetime.fromtimestamp(1483579888)
#priceTimeAmount = {}
#previousTime = datetime.strptime(date +' 15:00:00','%Y-%m-%d %H:%M:%S').timestamp()
#for index, row in df.iterrows():
#    if priceTimeAmount.get(row['price']) == None:
#        priceTimeAmount[row['price']] = { 'time':previousTime - row['time'], 'amount':row['amount'],'volume':row['volume'],'date':date}
#    else:
#        priceTimeAmount.get(row['price'])['time'] +=  previousTime - row['time']
#        priceTimeAmount.get(row['price'])['amount'] +=  row['amount']
#        priceTimeAmount.get(row['price'])['volume'] +=  row['volume']
#    
#    previousTime = row['time']
#    
#db = client.stock
#db.tickdata.insert(json.loads(df.to_json(orient='records')))

