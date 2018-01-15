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

num_of_days_a_sample = 16*16
tableName = 'SZ50VectorTrain'
#上证 50


def getTrainingExample(date,df):
    #trading days
    Num_Days = num_of_days_a_sample
    #Weeks
    days = Num_Days 
    df = df[df.index <= date].head(days)
    if len(df) < days:
        return None
    newdf = pd.DataFrame(data=[date],columns=['Date'])
    begin = False
    Dayiter = 1
    for index, row in df.iterrows():
        if index == date:
            begin = True
            composeADay(newdf,Dayiter,row)
            Dayiter+=1
            continue
        if begin and Dayiter <= Num_Days:
            composeADay(newdf,Dayiter,row)
            Dayiter+=1            

    newdf = newdf.drop(['Date'], axis=1)
    if len(newdf.columns) == (9 * (Num_Days)):
        return newdf
    else:
        return None

def getTradeInfoInRange(dfd):
    dfd = dfd.sort_index()
    newdf = pd.DataFrame(data=[1],columns=['ID'])
    newdf['OpenPrice'] = dfd['OpenPrice'].head(1)[0]
    newdf['ClosePrice'] = dfd['ClosePrice'].tail(1)[0]    
    newdf['LowPrice'] = dfd['LowPrice'].min()
    newdf['HighPrice'] = dfd['HighPrice'].max()
    newdf['Volume'] = dfd['Volume'].sum()
    newdf['Amount'] = dfd['Amount'].sum()
    #yesterday close - today's close
    newdf['Diff'] = dfd['ClosePrice'].head(1)[0] + dfd['Diff'].head(1)[0] - dfd['ClosePrice'].tail(1)[0]
    newdf['Percent'] = dfd['ClosePrice'].tail(1)[0] - (dfd['ClosePrice'].head(1)[0] + dfd['Diff'].head(1)[0]) / (dfd['ClosePrice'].head(1)[0] + dfd['Diff'].head(1)[0]) 
    newdf['Exchange'] = dfd['Exchange'].sum()
    newdf = newdf.drop(['ID'], axis=1)
    return newdf

def composeADay(df,iter,row):
    for columnName in row.index:
        df['Day'+str(iter)+columnName] = row[columnName]

def composeAWeek(df,iter,newdf):
    for columnName in newdf.columns:
        df['Week'+str(iter)+columnName] = newdf[columnName]

def importDataFromCSVToDB(workQueue):
    conn = pymysql.Connect(host="localhost",
                           port=3306,
                           user="root",
                           password="Initial0",
                           database="stock",
                           charset="utf8")
    cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)    
    tableCreated = False
    while(not workQueue.empty()):
        code = workQueue.get()
        #sql = "SELECT Date, OpenPrice,ClosePrice,Diff,Percent,LowPrice, HighPrice,Volume, Amount,Exchange FROM DAY where CODE = %s" 
        #cursor.execute(sql, (code,))
        #
        #table_rows = cursor.fetchall()
        df = pd.read_sql("SELECT Date, OpenPrice,ClosePrice,Diff,Percent,LowPrice, HighPrice,Volume, Amount,Exchange FROM DAY where CODE = %s", conn, index_col='Date',params=(code,),columns=['Date', 'OpenPrice','ClosePrice','Diff','Percent', 'LowPrice', 'HighPrice',  'Volume', 'Amount','Exchange'])
        #df.set_index(['Date'])
        df = df.sort_index(ascending=False)
        #484 is 9 years' data, 60 days is one training example length
        #ten days buffer
        tailTarget = min(242*9 + num_of_days_a_sample, len(df))
        df = df.iloc[0:tailTarget]
        maxLines = df.index.get_loc(df.index.min())
        if maxLines < num_of_days_a_sample:
            continue
        for index, row in df.iterrows():
            currentRowNum = df.index.get_loc(index)
            if currentRowNum > 0 and maxLines - currentRowNum > num_of_days_a_sample:
                newdf = df.iloc[currentRowNum:currentRowNum + num_of_days_a_sample]
                AExample = getTrainingExample(index,newdf)
                if AExample is None:
                    break
                row = df.iloc[currentRowNum]
                nextRowNum = currentRowNum - 1
                nextRow = df.iloc[nextRowNum]
                
                highPricePercent = (nextRow['HighPrice'] - row['ClosePrice']) / row['ClosePrice'] * 100
                lowPricePercent = (nextRow['LowPrice'] - row['ClosePrice']) / row['ClosePrice'] * 100
                AExample['NextDayHighPriceIncrease'] =  highPricePercent > 0
                AExample['NextDayHighPriceIncrease1'] = highPricePercent > 1
                AExample['NextDayHighPriceIncrease2'] = highPricePercent > 2
                AExample['NextDayHighPriceIncrease3'] = highPricePercent > 3
                AExample['NextDayHighPriceIncrease4'] = highPricePercent > 4
                AExample['NextDayHighPriceIncrease5'] = highPricePercent > 5
                AExample['NextDayHighPriceIncrease6'] = highPricePercent > 6                                                                                                
                AExample['NextDayHighPriceIncrease7'] = highPricePercent > 7
                AExample['NextDayHighPriceIncrease8'] = highPricePercent > 8
                AExample['NextDayHighPriceIncrease9'] = highPricePercent > 9
                AExample['NextDayHighPriceIncrease10'] =highPricePercent  > 9.9                
                AExample['NextDayHighPriceDecrease1'] = lowPricePercent< -1
                AExample['NextDayHighPriceDecrease2'] = lowPricePercent< -2
                AExample['NextDayHighPriceDecrease3'] = lowPricePercent< -3
                AExample['NextDayHighPriceDecrease4'] = lowPricePercent< -4
                AExample['NextDayHighPriceDecrease5'] = lowPricePercent< -5
                AExample['NextDayHighPriceDecrease6'] = lowPricePercent< -6
                AExample['NextDayHighPriceDecrease7'] = lowPricePercent< -7
                AExample['NextDayHighPriceDecrease8'] = lowPricePercent< -8
                AExample['NextDayHighPriceDecrease9'] = lowPricePercent< -9
                AExample['NextDayHighPriceDecrease10'] =lowPricePercent  < -9.9
                                 
                highPricePercent = (nextRow['ClosePrice'] - row['ClosePrice']) / row['ClosePrice'] * 100
                lowPricePercent = (nextRow['ClosePrice'] - row['ClosePrice']) / row['ClosePrice'] * 100
                AExample['NextDayClosePriceIncrease'] =  highPricePercent > 0
                AExample['NextDayClosePriceIncrease1'] = highPricePercent > 1
                AExample['NextDayClosePriceIncrease2'] = highPricePercent > 2
                AExample['NextDayClosePriceIncrease3'] = highPricePercent > 3
                AExample['NextDayClosePriceIncrease4'] = highPricePercent > 4
                AExample['NextDayClosePriceIncrease5'] = highPricePercent > 5
                AExample['NextDayClosePriceIncrease6'] = highPricePercent > 6                                                                                                
                AExample['NextDayClosePriceIncrease7'] = highPricePercent > 7
                AExample['NextDayClosePriceIncrease8'] = highPricePercent > 8
                AExample['NextDayClosePriceIncrease9'] = highPricePercent > 9
                AExample['NextDayClosePriceIncrease10'] =highPricePercent  > 9.9                
                AExample['NextDayClosePriceDecrease1'] = lowPricePercent< -1
                AExample['NextDayClosePriceDecrease2'] = lowPricePercent< -2
                AExample['NextDayClosePriceDecrease3'] = lowPricePercent< -3
                AExample['NextDayClosePriceDecrease4'] = lowPricePercent< -4
                AExample['NextDayClosePriceDecrease5'] = lowPricePercent< -5
                AExample['NextDayClosePriceDecrease6'] = lowPricePercent< -6
                AExample['NextDayClosePriceDecrease7'] = lowPricePercent< -7
                AExample['NextDayClosePriceDecrease8'] = lowPricePercent< -8
                AExample['NextDayClosePriceDecrease9'] = lowPricePercent< -9
                AExample['NextDayClosePriceDecrease10'] =lowPricePercent  < -9.9
                
                if currentRowNum >2:
                    regin = df.iloc[currentRowNum - 3 : currentRowNum]                    
                    highPricePercent = (regin['HighPrice'].max() - row['ClosePrice']) / row['ClosePrice'] * 100
                    lowPricePercent = (regin['LowPrice'].min() - row['ClosePrice']) / row['ClosePrice'] * 100
                    
                    AExample['Next3DayHighPriceIncrease'] =  highPricePercent >  0
                    AExample['Next3DayHighPriceIncrease1'] = highPricePercent >  1                    
                    AExample['Next3DayHighPriceIncrease3'] = highPricePercent >  3
                    AExample['Next3DayHighPriceIncrease5'] = highPricePercent >  5
                    AExample['Next3DayHighPriceIncrease7'] = highPricePercent >  7
                    AExample['Next3DayHighPriceIncrease9'] = highPricePercent >  9
                    AExample['Next3DayHighPriceIncrease11'] = highPricePercent > 11
                    AExample['Next3DayHighPriceIncrease13'] = highPricePercent > 13                                                                                                
                    AExample['Next3DayHighPriceIncrease15'] = highPricePercent > 15
                    AExample['Next3DayHighPriceIncrease17'] = highPricePercent > 17
                    AExample['Next3DayHighPriceIncrease20'] = highPricePercent > 20
                    AExample['Next3DayHighPriceIncrease25'] =highPricePercent >  25                
                    AExample['Next3DayHighPriceDecrease3'] = lowPricePercent< -3 
                    AExample['Next3DayHighPriceDecrease5'] = lowPricePercent< -5 
                    AExample['Next3DayHighPriceDecrease7'] = lowPricePercent< -7 
                    AExample['Next3DayHighPriceDecrease9'] = lowPricePercent< -9 
                    AExample['Next3DayHighPriceDecrease11'] = lowPricePercent< -11 
                    AExample['Next3DayHighPriceDecrease13'] = lowPricePercent< -13
                    AExample['Next3DayHighPriceDecrease15'] = lowPricePercent< -15
                    AExample['Next3DayHighPriceDecrease17'] = lowPricePercent< -17
                    AExample['Next3DayHighPriceDecrease20'] = lowPricePercent< -20
                    AExample['Next3DayHighPriceDecrease25'] =lowPricePercent <-25
                                                                               
                    highPricePercent = (regin['ClosePrice'].max() - row['ClosePrice']) / row['ClosePrice'] * 100
                    lowPricePercent = (regin['ClosePrice'].min() - row['ClosePrice']) / row['ClosePrice'] * 100
                    AExample['Next3DayClosePriceIncrease'] =  highPricePercent > 0 
                    AExample['Next3DayClosePriceIncrease1'] = highPricePercent > 1                     
                    AExample['Next3DayClosePriceIncrease3'] = highPricePercent > 3 
                    AExample['Next3DayClosePriceIncrease5'] = highPricePercent > 5 
                    AExample['Next3DayClosePriceIncrease7'] = highPricePercent > 7 
                    AExample['Next3DayClosePriceIncrease9'] = highPricePercent > 9 
                    AExample['Next3DayClosePriceIncrease11'] = highPricePercent > 11
                    AExample['Next3DayClosePriceIncrease13'] = highPricePercent > 13                                                                                             
                    AExample['Next3DayClosePriceIncrease15'] = highPricePercent > 15
                    AExample['Next3DayClosePriceIncrease17'] = highPricePercent > 17
                    AExample['Next3DayClosePriceIncrease20'] = highPricePercent > 20
                    AExample['Next3DayClosePriceIncrease25'] =highPricePercent  >25               
                    AExample['Next3DayClosePriceDecrease3'] = lowPricePercent< -3  
                    AExample['Next3DayClosePriceDecrease5'] = lowPricePercent< -5  
                    AExample['Next3DayClosePriceDecrease7'] = lowPricePercent< -7  
                    AExample['Next3DayClosePriceDecrease9'] = lowPricePercent< -9  
                    AExample['Next3DayClosePriceDecrease11'] = lowPricePercent<-11
                    AExample['Next3DayClosePriceDecrease13'] = lowPricePercent<-13
                    AExample['Next3DayClosePriceDecrease15'] = lowPricePercent<-15
                    AExample['Next3DayClosePriceDecrease17'] = lowPricePercent<-17
                    AExample['Next3DayClosePriceDecrease20'] = lowPricePercent<-20
                    AExample['Next3DayClosePriceDecrease25'] = lowPricePercent< -25 
                if currentRowNum >4:
                    regin = df.iloc[currentRowNum - 5 : currentRowNum]                    
                    highPricePercent = (regin['HighPrice'].max() - row['ClosePrice']) / row['ClosePrice'] * 100
                    lowPricePercent = (regin['LowPrice'].min() - row['ClosePrice']) / row['ClosePrice'] * 100
                    
                    AExample['Next5DayHighPriceIncrease'] =  highPricePercent >  0
                    AExample['Next5DayHighPriceIncrease1'] = highPricePercent >  1                    
                    AExample['Next5DayHighPriceIncrease3'] = highPricePercent >  3
                    AExample['Next5DayHighPriceIncrease5'] = highPricePercent >  5
                    AExample['Next5DayHighPriceIncrease7'] = highPricePercent >  7
                    AExample['Next5DayHighPriceIncrease9'] = highPricePercent >  9
                    AExample['Next5DayHighPriceIncrease11'] = highPricePercent > 11
                    AExample['Next5DayHighPriceIncrease13'] = highPricePercent > 13                                                                                                
                    AExample['Next5DayHighPriceIncrease15'] = highPricePercent > 15
                    AExample['Next5DayHighPriceIncrease17'] = highPricePercent > 17
                    AExample['Next5DayHighPriceIncrease20'] = highPricePercent > 20
                    AExample['Next5DayHighPriceIncrease25'] =highPricePercent >  25                
                    AExample['Next5DayHighPriceDecrease3'] = lowPricePercent< -3 
                    AExample['Next5DayHighPriceDecrease5'] = lowPricePercent< -5 
                    AExample['Next5DayHighPriceDecrease7'] = lowPricePercent< -7 
                    AExample['Next5DayHighPriceDecrease9'] = lowPricePercent< -9 
                    AExample['Next5DayHighPriceDecrease11'] = lowPricePercent< -11 
                    AExample['Next5DayHighPriceDecrease13'] = lowPricePercent< -13
                    AExample['Next5DayHighPriceDecrease15'] = lowPricePercent< -15
                    AExample['Next5DayHighPriceDecrease17'] = lowPricePercent< -17
                    AExample['Next5DayHighPriceDecrease20'] = lowPricePercent< -20
                    AExample['Next5DayHighPriceDecrease25'] =lowPricePercent <-25
                                                                               
                    highPricePercent = (regin['ClosePrice'].max() - row['ClosePrice']) / row['ClosePrice'] * 100
                    lowPricePercent = (regin['ClosePrice'].min() - row['ClosePrice']) / row['ClosePrice'] * 100
                    AExample['Next5DayClosePriceIncrease'] =  highPricePercent > 0 
                    AExample['Next5DayClosePriceIncrease1'] = highPricePercent > 1 
                    AExample['Next5DayClosePriceIncrease5'] = highPricePercent > 5 
                    AExample['Next5DayClosePriceIncrease7'] = highPricePercent > 7 
                    AExample['Next5DayClosePriceIncrease9'] = highPricePercent > 9 
                    AExample['Next5DayClosePriceIncrease11'] = highPricePercent > 11
                    AExample['Next5DayClosePriceIncrease13'] = highPricePercent > 13                                                                                             
                    AExample['Next5DayClosePriceIncrease15'] = highPricePercent > 15
                    AExample['Next5DayClosePriceIncrease17'] = highPricePercent > 17
                    AExample['Next5DayClosePriceIncrease20'] = highPricePercent > 20
                    AExample['Next5DayClosePriceIncrease25'] =highPricePercent  >25               
                    AExample['Next5DayClosePriceDecrease3'] = lowPricePercent< -3  
                    AExample['Next5DayClosePriceDecrease5'] = lowPricePercent< -5  
                    AExample['Next5DayClosePriceDecrease7'] = lowPricePercent< -7  
                    AExample['Next5DayClosePriceDecrease9'] = lowPricePercent< -9  
                    AExample['Next5DayClosePriceDecrease11'] = lowPricePercent<-11
                    AExample['Next5DayClosePriceDecrease13'] = lowPricePercent<-13
                    AExample['Next5DayClosePriceDecrease15'] = lowPricePercent<-15
                    AExample['Next5DayClosePriceDecrease17'] = lowPricePercent<-17
                    AExample['Next5DayClosePriceDecrease20'] = lowPricePercent<-20
                    AExample['Next5DayClosePriceDecrease25'] = lowPricePercent< -25                     
                #getColumns(AExample.iloc[0])
                if not tableCreated:
                    tableCreated = createTable(AExample,tableName,cursor)
                
                insertAExample(AExample.iloc[0],cursor,conn,index,code)
                                                                                                                                  
    cursor.close()
    conn.close()
    
def getColumns(row):
    names = ''
    for columnName in row.index:
        names += ','+columnName
    return names

from datetime import date,datetime
def insertAExample(row,cursor,conn,Date,Code):
    prefix = 'insert into {} (DateString,Date,code'.format(tableName)
    surfix = ') values(%s,%s'
    stockdate = datetime.strptime(Date,'%Y-%m-%d')
    stockDateLong = stockdate.timestamp()
    values = [str(Date),str(stockDateLong),str(Code)]
    for columnName in row.index:
        if(isinstance(row[columnName],np.float64)):
            values.append(str(row[columnName]))
        else:
            if row[columnName]:
                values.append(1)
            else:
                values.append(0)
        prefix+=','+columnName
        surfix+=',%s'
    sql = prefix + surfix + ')'
    
    cursor.execute(sql,values)
    conn.commit()
    

def createTable(df, tableName,cursor):
    try:
        cursor.execute('SELECT 1 FROM '+tableName+' LIMIT 1;')
    except Exception:
        prefix = 'CREATE TABLE `'+tableName+ '` ( `id` INT NOT NULL AUTO_INCREMENT,'
        surfix = '  PRIMARY KEY (`id`),  UNIQUE INDEX `id_UNIQUE` (`id` ASC));'
        sql=prefix
        sql += '`DateString` VARCHAR(45) NOT NULL,'
        sql += '`Date` DECIMAL(18) NOT NULL,' 
        sql += '`Code` DECIMAL(18,3) NOT NULL,'         
        for columnName in df.columns:
            if columnName.startswith('Next'):
                sql += '`'+columnName+'`tinyint(1),'
            else:
                sql += '`'+columnName+'` DECIMAL(18,3) NOT NULL,'
        sql +=surfix
        
        cursor.execute(sql)
        #
    return True;

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

    codes = ['600000','600016','600028','600029','600030','600036','600048','600050','600100','600104','600109','600111','600485','600518','600519','600547','600637','600837','600887','600893','600958','600999','601006','601088','601166','601169','601186','601198','601211','601288','601318','601328','601336','601377','601390','601398','601601','601628','601668','601688','601766','601788','601800','601818','601857','601901','601985','601988','601989','601998','600016','600015','600036','600000','601166','601398','601328','601939','601169','601288','000001','601009','601988','601818','002142','601998','600919','601997','002807']
    for code in codes:
            workQueue.put(code)
    
    cursor.close()
    conn.close()
    
    for i in range(1):
        t = threading.Thread(target=importDataFromCSVToDB, args=(workQueue,))
        t.start()
        time.sleep(random.randint(low=0, high=10))
