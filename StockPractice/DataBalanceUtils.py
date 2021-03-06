import sys
sys.path.append('..')

from mxnet import gluon
from mxnet import ndarray as nd
import pandas as pd
import numpy as np
import pymysql
import threading
from numpy import random
import time
from multiprocessing import Queue

tableName = 'SZ50Bank5D3'
predictCategory = 'NextDayIncrease1'
copyNumbers = 1

if len(sys.argv) == 4:
    tableName = sys.argv[1] #'banktraining'
    predictCategory = sys.argv[2] #'NextDayIncrease1'
    copyNumbers = int(sys.argv[3]) #copyNumbers

def getSQLColumnString(columns):
    if len(columns) < 1:
        return
    select = columns[0]
    for name in columns[1:]:
       select += ","+name
    return select

def insertAExample(row,cursor,conn):
    prefix = 'insert into {}('.format(tableName)
    surfix = ') values('
    values = []
    first = True
    for columnName in row.index:
        if(isinstance(row[columnName],np.float64)):
            values.append(str(row[columnName]))
        elif(isinstance(row[columnName],float)):
            values.append(str(row[columnName]))            
        elif isinstance(row[columnName],str):
            values.append(str(row[columnName]))
        elif isinstance(row[columnName],int):
            values.append(str(row[columnName]))        
        else:
            if row[columnName]:
                values.append(1)
            else:
                values.append(0)
        if first:
            prefix+=columnName
            surfix+='%s'
            first = False
        else:
            prefix+=','+columnName
            surfix+=',%s'
    sql = prefix + surfix + ')'
    
    cursor.execute(sql,values)
    conn.commit()

def databalanceGenerator(workQueue,maxdatabalance):
    conn = pymysql.Connect(host="localhost",
                           port=3306,
                           user="root",
                           password="Initial0",
                           database="stock",
                           charset="utf8")
    
    cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)   
    featureNames = ['DateString','Date','Code','Day1OpenPrice','Day1ClosePrice','Day1Diff','Day1Percent','Day1LowPrice','Day1HighPrice','Day1Volume','Day1Amount','Day1Exchange','Day2OpenPrice','Day2ClosePrice','Day2Diff','Day2Percent','Day2LowPrice','Day2HighPrice','Day2Volume','Day2Amount','Day2Exchange','Day3OpenPrice','Day3ClosePrice','Day3Diff','Day3Percent','Day3LowPrice','Day3HighPrice','Day3Volume','Day3Amount','Day3Exchange','Day4OpenPrice','Day4ClosePrice','Day4Diff','Day4Percent','Day4LowPrice','Day4HighPrice','Day4Volume','Day4Amount','Day4Exchange','Day5OpenPrice','Day5ClosePrice','Day5Diff','Day5Percent','Day5LowPrice','Day5HighPrice','Day5Volume','Day5Amount','Day5Exchange','Day6OpenPrice','Day6ClosePrice','Day6Diff','Day6Percent','Day6LowPrice','Day6HighPrice','Day6Volume','Day6Amount','Day6Exchange','Day7OpenPrice','Day7ClosePrice','Day7Diff','Day7Percent','Day7LowPrice','Day7HighPrice','Day7Volume','Day7Amount','Day7Exchange','Day8OpenPrice','Day8ClosePrice','Day8Diff','Day8Percent','Day8LowPrice','Day8HighPrice','Day8Volume','Day8Amount','Day8Exchange','Day9OpenPrice','Day9ClosePrice','Day9Diff','Day9Percent','Day9LowPrice','Day9HighPrice','Day9Volume','Day9Amount','Day9Exchange','Day10OpenPrice','Day10ClosePrice','Day10Diff','Day10Percent','Day10LowPrice','Day10HighPrice','Day10Volume','Day10Amount','Day10Exchange','Day11OpenPrice','Day11ClosePrice','Day11Diff','Day11Percent','Day11LowPrice','Day11HighPrice','Day11Volume','Day11Amount','Day11Exchange','Day12OpenPrice','Day12ClosePrice','Day12Diff','Day12Percent','Day12LowPrice','Day12HighPrice','Day12Volume','Day12Amount','Day12Exchange','Day13OpenPrice','Day13ClosePrice','Day13Diff','Day13Percent','Day13LowPrice','Day13HighPrice','Day13Volume','Day13Amount','Day13Exchange','Day14OpenPrice','Day14ClosePrice','Day14Diff','Day14Percent','Day14LowPrice','Day14HighPrice','Day14Volume','Day14Amount','Day14Exchange','Day15OpenPrice','Day15ClosePrice','Day15Diff','Day15Percent','Day15LowPrice','Day15HighPrice','Day15Volume','Day15Amount','Day15Exchange','Day16OpenPrice','Day16ClosePrice','Day16Diff','Day16Percent','Day16LowPrice','Day16HighPrice','Day16Volume','Day16Amount','Day16Exchange','Day17OpenPrice','Day17ClosePrice','Day17Diff','Day17Percent','Day17LowPrice','Day17HighPrice','Day17Volume','Day17Amount','Day17Exchange','Day18OpenPrice','Day18ClosePrice','Day18Diff','Day18Percent','Day18LowPrice','Day18HighPrice','Day18Volume','Day18Amount','Day18Exchange','Day19OpenPrice','Day19ClosePrice','Day19Diff','Day19Percent','Day19LowPrice','Day19HighPrice','Day19Volume','Day19Amount','Day19Exchange','Day20OpenPrice','Day20ClosePrice','Day20Diff','Day20Percent','Day20LowPrice','Day20HighPrice','Day20Volume','Day20Amount','Day20Exchange','Day21OpenPrice','Day21ClosePrice','Day21Diff','Day21Percent','Day21LowPrice','Day21HighPrice','Day21Volume','Day21Amount','Day21Exchange','Day22OpenPrice','Day22ClosePrice','Day22Diff','Day22Percent','Day22LowPrice','Day22HighPrice','Day22Volume','Day22Amount','Day22Exchange','Day23OpenPrice','Day23ClosePrice','Day23Diff','Day23Percent','Day23LowPrice','Day23HighPrice','Day23Volume','Day23Amount','Day23Exchange','Day24OpenPrice','Day24ClosePrice','Day24Diff','Day24Percent','Day24LowPrice','Day24HighPrice','Day24Volume','Day24Amount','Day24Exchange','Day25OpenPrice','Day25ClosePrice','Day25Diff','Day25Percent','Day25LowPrice','Day25HighPrice','Day25Volume','Day25Amount','Day25Exchange','Day26OpenPrice','Day26ClosePrice','Day26Diff','Day26Percent','Day26LowPrice','Day26HighPrice','Day26Volume','Day26Amount','Day26Exchange','Day27OpenPrice','Day27ClosePrice','Day27Diff','Day27Percent','Day27LowPrice','Day27HighPrice','Day27Volume','Day27Amount','Day27Exchange','Day28OpenPrice','Day28ClosePrice','Day28Diff','Day28Percent','Day28LowPrice','Day28HighPrice','Day28Volume','Day28Amount','Day28Exchange','Day29OpenPrice','Day29ClosePrice','Day29Diff','Day29Percent','Day29LowPrice','Day29HighPrice','Day29Volume','Day29Amount','Day29Exchange','Day30OpenPrice','Day30ClosePrice','Day30Diff','Day30Percent','Day30LowPrice','Day30HighPrice','Day30Volume','Day30Amount','Day30Exchange','Day31OpenPrice','Day31ClosePrice','Day31Diff','Day31Percent','Day31LowPrice','Day31HighPrice','Day31Volume','Day31Amount','Day31Exchange','Day32OpenPrice','Day32ClosePrice','Day32Diff','Day32Percent','Day32LowPrice','Day32HighPrice','Day32Volume','Day32Amount','Day32Exchange','Day33OpenPrice','Day33ClosePrice','Day33Diff','Day33Percent','Day33LowPrice','Day33HighPrice','Day33Volume','Day33Amount','Day33Exchange','Day34OpenPrice','Day34ClosePrice','Day34Diff','Day34Percent','Day34LowPrice','Day34HighPrice','Day34Volume','Day34Amount','Day34Exchange','Day35OpenPrice','Day35ClosePrice','Day35Diff','Day35Percent','Day35LowPrice','Day35HighPrice','Day35Volume','Day35Amount','Day35Exchange','Day36OpenPrice','Day36ClosePrice','Day36Diff','Day36Percent','Day36LowPrice','Day36HighPrice','Day36Volume','Day36Amount','Day36Exchange','Day37OpenPrice','Day37ClosePrice','Day37Diff','Day37Percent','Day37LowPrice','Day37HighPrice','Day37Volume','Day37Amount','Day37Exchange','Day38OpenPrice','Day38ClosePrice','Day38Diff','Day38Percent','Day38LowPrice','Day38HighPrice','Day38Volume','Day38Amount','Day38Exchange','Day39OpenPrice','Day39ClosePrice','Day39Diff','Day39Percent','Day39LowPrice','Day39HighPrice','Day39Volume','Day39Amount','Day39Exchange','Day40OpenPrice','Day40ClosePrice','Day40Diff','Day40Percent','Day40LowPrice','Day40HighPrice','Day40Volume','Day40Amount','Day40Exchange','Week1OpenPrice','Week1ClosePrice','Week1LowPrice','Week1HighPrice','Week1Volume','Week1Amount','Week1Diff','Week1Percent','Week1Exchange','Week2OpenPrice','Week2ClosePrice','Week2LowPrice','Week2HighPrice','Week2Volume','Week2Amount','Week2Diff','Week2Percent','Week2Exchange','Week3OpenPrice','Week3ClosePrice','Week3LowPrice','Week3HighPrice','Week3Volume','Week3Amount','Week3Diff','Week3Percent','Week3Exchange','Week4OpenPrice','Week4ClosePrice','Week4LowPrice','Week4HighPrice','Week4Volume','Week4Amount','Week4Diff','Week4Percent','Week4Exchange']    
    predictNames = ['NextDayIncrease','NextDayIncrease1','NextDayIncrease2','NextDayIncrease3','NextDayIncrease4','NextDayIncrease5','NextDayIncrease6','NextDayIncrease7','NextDayIncrease8','NextDayIncrease9','NextDayIncrease10','NextDayDecrease1','NextDayDecrease2','NextDayDecrease3','NextDayDecrease4','NextDayDecrease5','NextDayDecrease6','NextDayDecrease7','NextDayDecrease8','NextDayDecrease9','NextDayDecrease10']
    
    while(not workQueue.empty()):
        code = workQueue.get()
        df = pd.read_sql("SELECT {} from {} where code = {} and DATABALANCE = -1 and {} = true ".format(getSQLColumnString(featureNames + predictNames),tableName,code,predictCategory), 
                         conn, 
                         columns=featureNames + predictNames)
        
        dfless = df[df[predictCategory] == 1]
        #unbalaced class data enhance, add more data through giving a random_normal to the real data
        enhanceIter = copyNumbers #len(df) // len(dfless)
        for i in range(enhanceIter):
            dflessCopy = dfless.copy(True)
            dflessCopy['DATABALANCE'] = (i+maxdatabalance+1)            
            for columnName in dfless.iloc[0].index:
                if not columnName.startswith('Next'):
                    normalarr = None
                    if columnName.__contains__('Percent') or columnName.__contains__('Exchange') or columnName.__contains__('Diff'):
                        normalarr = nd.random_normal(shape=(len(dfless), 1),scale=0.01).asnumpy()
                    elif columnName.__contains__('Price'):
                        normalarr = nd.random_normal(shape=(len(dfless), 1),scale=1).asnumpy()
                        #
                    elif isinstance(df[columnName].iloc[0],str) or columnName.__contains__('Code') or columnName.__contains__('CODELONG'):
                        #columnName.__contains__('Date') or columnName.__contains__('Code') or columnName.__contains__('CODELONG') 
                        continue
                    else:
                        normalarr = nd.random_normal(shape=(len(dfless), 1),scale=100).asnumpy()
                    dflessCopy[columnName] = dflessCopy[columnName].values.reshape(len(dfless),1) + normalarr
            dflessCopy['DATABALANCE'] = (i+maxdatabalance+1)
            for index, row in dflessCopy.iterrows():
                insertAExample(row,cursor,conn)
            #ignore_index true, otherwise duplicate index will happen
            #df = df.append(dflessCopy,ignore_index=True)
    
    cursor.close()
    conn.close()

workQueue = Queue(5000)

if __name__ == '__main__':    
    conn = pymysql.Connect(host="localhost",
                           port=3306,
                           user="root",
                           password="Initial0",
                           database="stock",
                           charset="utf8")
    cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
    
    cursor.execute('select MAX(DATABALANCE) AS MAXDATABALANCE from {}'.format(tableName))
    result = cursor.fetchall()
    result = pd.DataFrame(result)
    maxdatabalance = result['MAXDATABALANCE'][0]
    
    cursor.execute('SELECT code FROM stock.{} group by code'.format(tableName))
    result = cursor.fetchall()
    df = pd.DataFrame(result)
    codes = list(df['code'])
    for code in codes:
        workQueue.put(code)
        
    for i in range(8):
        t = threading.Thread(target=databalanceGenerator, args=(workQueue,maxdatabalance))
        t.start()
        time.sleep(random.randint(low=0, high=10))
    
    cursor.close()
    conn.close()





1