# coding=utf-8

import pandas as pd
import sys
import numpy as np
import pymysql
import os
import threading
from numpy import random
import time
from datetime import date, datetime 
from multiprocessing import Queue
import urllib
import zlib
import json
import requests
import traceback

pauseDate = '2017-10-20'
tableName = 'daytrainexample8'

def getTrainingExample(date,df):
    #trading days
    Num_Days = 40
    #Weeks
    Num_Weeks = 4
    days = Num_Days + Num_Weeks*5 
    df = df[df.index <= date].head(days)
    if len(df) < days:
        return None
    newdf = pd.DataFrame(data=[date],columns=['DateString'])
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
    newdf = newdf.drop(['DateString'], axis=1)
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
        if columnName != 'Date':
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
        df = pd.read_sql("SELECT DateString,Date, OpenPrice,ClosePrice,Diff,Percent,LowPrice, HighPrice,Volume, Amount,Exchange FROM DAY where CODE = %s and date > 1451577600", conn, index_col='DateString',params=(code,),columns=['DateString','Date', 'OpenPrice','ClosePrice','Diff','Percent', 'LowPrice', 'HighPrice',  'Volume', 'Amount','Exchange'])
        #df.set_index(['Date'])
        if len(df) == 0:
            break 
        df = df.sort_index(ascending=False)
        #484 is two days' data, 60 days is one training example length
        #ten days buffer
        tailTarget = min(242*5 + 60, len(df))
        df = df.iloc[0:tailTarget]
        maxLines = 0
        try:
            maxLines = df.index.get_loc(df.index.min())
        except Exception:
                print(sys.exc_info())
                print(traceback.print_exc())
            
        if maxLines < 60:
            continue
        for index, row in df.iterrows():
            currentRowNum = df.index.get_loc(index)
            if currentRowNum > 0 and maxLines - currentRowNum > 60:
                if datetime.strptime(index,'%Y-%m-%d').timestamp() > 1508428800:
                    continue;
                if datetime.strptime(index,'%Y-%m-%d').timestamp() < 1507824000:
                    break;                
                newdf = df.iloc[currentRowNum:currentRowNum + 60]
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
                    AExample['Next5DayClosePriceIncrease3'] = highPricePercent > 3 
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
                #if not tableCreated:
                #    tableCreated = createTable(AExample,'dayTrainExample8',cursor,conn)
                
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
    prefix = 'insert into {}(DateString,Date,code'.format(tableName)
    surfix = ') values(%s,%s,%s'
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
    

def createTable(df, tableName,cursor,conn):
    try:
        cursor.execute('SELECT 1 FROM '+tableName+' LIMIT 1;')
    except Exception:
        prefix = 'CREATE TABLE `'+tableName+ '` ( `id` INT NOT NULL AUTO_INCREMENT,'
        surfix = '  PRIMARY KEY (`id`),  UNIQUE INDEX `id_UNIQUE` (`id` ASC));'
        sql=prefix
        sql += '`Date` VARCHAR(45) NOT NULL,'
        sql += '`DateLong` DECIMAL(18) NOT NULL,' 
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
    
    cursor.execute('SELECT code FROM stock.Bank1PercentPredict group by code')
    result = cursor.fetchall()
    existed = []
    if len(result) != 0:
        result = pd.DataFrame(result)
        existed = set(result['code'])

    cursor.execute('SELECT code FROM stock.daytrainexample8 group by code')
    result = cursor.fetchall()
    codes = []
    if len(result) != 0:
        result = pd.DataFrame(result)
        codes = set(result['code'])
    for code in codes:
        workQueue.put(code)
    
    cursor.close()
    conn.close()
    
    for i in range(32):
        t = threading.Thread(target=importDataFromCSVToDB, args=(workQueue,))
        t.start()
        time.sleep(random.randint(low=0, high=10))