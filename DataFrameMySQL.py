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


def getTrainingExample(date,df):
    #trading days
    Num_Days = 40
    #Weeks
    Num_Weeks = 4
    days = Num_Days + Num_Weeks*5 
    df = df[df.index <= date].head(days)
    if len(df) < days:
        return None
    newdf = pd.DataFrame(data=[date],columns=['Date'])
    begin = False
    Dayiter = 1
    WeekStartLineNum = None
    WeekIter=1
    for index, row in df.iterrows():
        if index == date:
            begin = True
            composeADay(newdf,Dayiter,row)
            Dayiter+=1
            continue
        if begin and Dayiter <= Num_Days:
            composeADay(newdf,Dayiter,row)
            if(Dayiter == Num_Days):
                WeekStartLineNum = df.index.get_loc(index)
            Dayiter+=1
        if WeekStartLineNum != None and (df.index.get_loc(index) - WeekStartLineNum) % 5 ==1:
            lineNum = df.index.get_loc(index)
            indexRange = []
            for i in range(5):
                indexRange.append(df.index[lineNum+i])
            dfd = df[df.index.isin(indexRange)]
            weekdf = getTradeInfoInRange(dfd)
            composeAWeek(newdf,WeekIter,weekdf)
            if(WeekIter == Num_Weeks):
                break
            WeekIter+=1
    newdf = newdf.drop(['Date'], axis=1)
    if len(newdf.columns) == (9 * (Num_Days+Num_Weeks)):
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
        #484 is two days' data, 60 days is one training example length
        tailTarget = min(485 + 60, len(df))
        df = df.iloc[0:tailTarget]
        maxLines = df.index.get_loc(df.index.min())
        if maxLines < 60:
            continue
        for index, row in df.iterrows():
            currentRowNum = df.index.get_loc(index)
            if currentRowNum > 0 and maxLines - currentRowNum > 60:
                newdf = df.iloc[currentRowNum:currentRowNum + 60]
                AExample = getTrainingExample(index,newdf)
                if AExample is None:
                    break
                row = df.iloc[currentRowNum]
                nextRowNum = currentRowNum - 1
                nextRow = df.iloc[nextRowNum]
                highPricePercent = (nextRow['HighPrice'] - row['ClosePrice']) / row['ClosePrice'] * 100
                AExample['NextDayIncrease'] =  highPricePercent > 0
                AExample['NextDayIncrease1'] = highPricePercent > 1
                AExample['NextDayIncrease2'] = highPricePercent > 2
                AExample['NextDayIncrease3'] = highPricePercent > 3
                AExample['NextDayIncrease4'] = highPricePercent > 4
                AExample['NextDayIncrease5'] = highPricePercent > 5
                AExample['NextDayIncrease6'] = highPricePercent > 6                                                                                                
                AExample['NextDayIncrease7'] = highPricePercent > 7
                AExample['NextDayIncrease8'] = highPricePercent > 8
                AExample['NextDayIncrease9'] = highPricePercent > 9
                AExample['NextDayIncrease10'] =highPricePercent  > 9.9
                
                lowPricePercent = (nextRow['LowPrice'] - row['ClosePrice']) / row['ClosePrice'] * 100
                AExample['NextDayDecrease1'] = lowPricePercent< -1
                AExample['NextDayDecrease2'] = lowPricePercent< -2
                AExample['NextDayDecrease3'] = lowPricePercent< -3
                AExample['NextDayDecrease4'] = lowPricePercent< -4
                AExample['NextDayDecrease5'] = lowPricePercent< -5
                AExample['NextDayDecrease6'] = lowPricePercent< -6
                AExample['NextDayDecrease7'] = lowPricePercent< -7
                AExample['NextDayDecrease8'] = lowPricePercent< -8
                AExample['NextDayDecrease9'] = lowPricePercent< -9
                AExample['NextDayDecrease10'] =lowPricePercent  < -9.9

                getColumns(AExample.iloc[0])
                if not tableCreated:
                    tableCreated = createTable(AExample,'dayTrainExample',cursor,conn)
                
                insertAExample(AExample.iloc[0],cursor,conn,index,code)
                                                                                                                                  
    cursor.close()
    conn.close()
    
def getColumns(row):
    names = ''
    for columnName in row.index:
        names += ','+columnName
    return names

def insertAExample(row,cursor,conn,Date,Code):
    prefix = 'insert into dayTrainExample(Date,code'
    surfix = ') values(%s,%s'
    values = [str(Date),str(Code)]
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
    

def createTable(df, tableName,cursor,conn):
    try:
        cursor.execute('SELECT 1 FROM '+tableName+' LIMIT 1;')
    except Exception:
        prefix = 'CREATE TABLE `'+tableName+ '` ( `id` INT NOT NULL AUTO_INCREMENT,'
        surfix = '  PRIMARY KEY (`id`),  UNIQUE INDEX `id_UNIQUE` (`id` ASC));'
        sql=prefix
        sql += '`Code` VARCHAR(45) NOT NULL,' 
        sql += '`Date` VARCHAR(45) NOT NULL,' 
        for columnName in df.columns:
            if columnName[0:6] == 'NextDay':
                sql += '`'+columnName+'`tinyint(1) NOT NULL,'
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

    #cursor.execute('select code from (SELECT code,count(*) as numbers FROM stock.day group by code ) as counttable where numbers > 60')
    #result = cursor.fetchall()
    #existed = pd.DataFrame(result)
    existed =set(['600016']) #set(existed['code'])
    for code in existed:
        workQueue.put(code)
    
    cursor.close()
    conn.close()
    
    for i in range(10):
        t = threading.Thread(target=importDataFromCSVToDB, args=(workQueue,))
        t.start()
        time.sleep(random.randint(low=0, high=10))
