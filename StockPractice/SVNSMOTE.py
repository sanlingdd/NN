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
from imblearn.over_sampling import SMOTE, ADASYN

tableName = 'SZ50Bank5D3'
predictCategory = 'Next5DayClosePriceIncrease3'

def getSQLColumnString(columns):
    if len(columns) < 1:
        return
    select = columns[0]
    for name in columns[1:]:
       select += ","+name
    return select

conn = pymysql.Connect(host="localhost",
                       port=3306,
                       user="root",
                       password="Initial0",
                       database="stock",
                       charset="utf8")

cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)   
featureNames = ['Date','Code','Day1OpenPrice','Day1ClosePrice','Day1Diff','Day1Percent','Day1LowPrice','Day1HighPrice','Day1Volume','Day1Amount','Day1Exchange','Day2OpenPrice','Day2ClosePrice','Day2Diff','Day2Percent','Day2LowPrice','Day2HighPrice','Day2Volume','Day2Amount','Day2Exchange','Day3OpenPrice','Day3ClosePrice','Day3Diff','Day3Percent','Day3LowPrice','Day3HighPrice','Day3Volume','Day3Amount','Day3Exchange','Day4OpenPrice','Day4ClosePrice','Day4Diff','Day4Percent','Day4LowPrice','Day4HighPrice','Day4Volume','Day4Amount','Day4Exchange','Day5OpenPrice','Day5ClosePrice','Day5Diff','Day5Percent','Day5LowPrice','Day5HighPrice','Day5Volume','Day5Amount','Day5Exchange','Day6OpenPrice','Day6ClosePrice','Day6Diff','Day6Percent','Day6LowPrice','Day6HighPrice','Day6Volume','Day6Amount','Day6Exchange','Day7OpenPrice','Day7ClosePrice','Day7Diff','Day7Percent','Day7LowPrice','Day7HighPrice','Day7Volume','Day7Amount','Day7Exchange','Day8OpenPrice','Day8ClosePrice','Day8Diff','Day8Percent','Day8LowPrice','Day8HighPrice','Day8Volume','Day8Amount','Day8Exchange','Day9OpenPrice','Day9ClosePrice','Day9Diff','Day9Percent','Day9LowPrice','Day9HighPrice','Day9Volume','Day9Amount','Day9Exchange','Day10OpenPrice','Day10ClosePrice','Day10Diff','Day10Percent','Day10LowPrice','Day10HighPrice','Day10Volume','Day10Amount','Day10Exchange','Day11OpenPrice','Day11ClosePrice','Day11Diff','Day11Percent','Day11LowPrice','Day11HighPrice','Day11Volume','Day11Amount','Day11Exchange','Day12OpenPrice','Day12ClosePrice','Day12Diff','Day12Percent','Day12LowPrice','Day12HighPrice','Day12Volume','Day12Amount','Day12Exchange','Day13OpenPrice','Day13ClosePrice','Day13Diff','Day13Percent','Day13LowPrice','Day13HighPrice','Day13Volume','Day13Amount','Day13Exchange','Day14OpenPrice','Day14ClosePrice','Day14Diff','Day14Percent','Day14LowPrice','Day14HighPrice','Day14Volume','Day14Amount','Day14Exchange','Day15OpenPrice','Day15ClosePrice','Day15Diff','Day15Percent','Day15LowPrice','Day15HighPrice','Day15Volume','Day15Amount','Day15Exchange','Day16OpenPrice','Day16ClosePrice','Day16Diff','Day16Percent','Day16LowPrice','Day16HighPrice','Day16Volume','Day16Amount','Day16Exchange','Day17OpenPrice','Day17ClosePrice','Day17Diff','Day17Percent','Day17LowPrice','Day17HighPrice','Day17Volume','Day17Amount','Day17Exchange','Day18OpenPrice','Day18ClosePrice','Day18Diff','Day18Percent','Day18LowPrice','Day18HighPrice','Day18Volume','Day18Amount','Day18Exchange','Day19OpenPrice','Day19ClosePrice','Day19Diff','Day19Percent','Day19LowPrice','Day19HighPrice','Day19Volume','Day19Amount','Day19Exchange','Day20OpenPrice','Day20ClosePrice','Day20Diff','Day20Percent','Day20LowPrice','Day20HighPrice','Day20Volume','Day20Amount','Day20Exchange','Day21OpenPrice','Day21ClosePrice','Day21Diff','Day21Percent','Day21LowPrice','Day21HighPrice','Day21Volume','Day21Amount','Day21Exchange','Day22OpenPrice','Day22ClosePrice','Day22Diff','Day22Percent','Day22LowPrice','Day22HighPrice','Day22Volume','Day22Amount','Day22Exchange','Day23OpenPrice','Day23ClosePrice','Day23Diff','Day23Percent','Day23LowPrice','Day23HighPrice','Day23Volume','Day23Amount','Day23Exchange','Day24OpenPrice','Day24ClosePrice','Day24Diff','Day24Percent','Day24LowPrice','Day24HighPrice','Day24Volume','Day24Amount','Day24Exchange','Day25OpenPrice','Day25ClosePrice','Day25Diff','Day25Percent','Day25LowPrice','Day25HighPrice','Day25Volume','Day25Amount','Day25Exchange','Day26OpenPrice','Day26ClosePrice','Day26Diff','Day26Percent','Day26LowPrice','Day26HighPrice','Day26Volume','Day26Amount','Day26Exchange','Day27OpenPrice','Day27ClosePrice','Day27Diff','Day27Percent','Day27LowPrice','Day27HighPrice','Day27Volume','Day27Amount','Day27Exchange','Day28OpenPrice','Day28ClosePrice','Day28Diff','Day28Percent','Day28LowPrice','Day28HighPrice','Day28Volume','Day28Amount','Day28Exchange','Day29OpenPrice','Day29ClosePrice','Day29Diff','Day29Percent','Day29LowPrice','Day29HighPrice','Day29Volume','Day29Amount','Day29Exchange','Day30OpenPrice','Day30ClosePrice','Day30Diff','Day30Percent','Day30LowPrice','Day30HighPrice','Day30Volume','Day30Amount','Day30Exchange','Day31OpenPrice','Day31ClosePrice','Day31Diff','Day31Percent','Day31LowPrice','Day31HighPrice','Day31Volume','Day31Amount','Day31Exchange','Day32OpenPrice','Day32ClosePrice','Day32Diff','Day32Percent','Day32LowPrice','Day32HighPrice','Day32Volume','Day32Amount','Day32Exchange','Day33OpenPrice','Day33ClosePrice','Day33Diff','Day33Percent','Day33LowPrice','Day33HighPrice','Day33Volume','Day33Amount','Day33Exchange','Day34OpenPrice','Day34ClosePrice','Day34Diff','Day34Percent','Day34LowPrice','Day34HighPrice','Day34Volume','Day34Amount','Day34Exchange','Day35OpenPrice','Day35ClosePrice','Day35Diff','Day35Percent','Day35LowPrice','Day35HighPrice','Day35Volume','Day35Amount','Day35Exchange','Day36OpenPrice','Day36ClosePrice','Day36Diff','Day36Percent','Day36LowPrice','Day36HighPrice','Day36Volume','Day36Amount','Day36Exchange','Day37OpenPrice','Day37ClosePrice','Day37Diff','Day37Percent','Day37LowPrice','Day37HighPrice','Day37Volume','Day37Amount','Day37Exchange','Day38OpenPrice','Day38ClosePrice','Day38Diff','Day38Percent','Day38LowPrice','Day38HighPrice','Day38Volume','Day38Amount','Day38Exchange','Day39OpenPrice','Day39ClosePrice','Day39Diff','Day39Percent','Day39LowPrice','Day39HighPrice','Day39Volume','Day39Amount','Day39Exchange','Day40OpenPrice','Day40ClosePrice','Day40Diff','Day40Percent','Day40LowPrice','Day40HighPrice','Day40Volume','Day40Amount','Day40Exchange','Week1OpenPrice','Week1ClosePrice','Week1LowPrice','Week1HighPrice','Week1Volume','Week1Amount','Week1Diff','Week1Percent','Week1Exchange','Week2OpenPrice','Week2ClosePrice','Week2LowPrice','Week2HighPrice','Week2Volume','Week2Amount','Week2Diff','Week2Percent','Week2Exchange','Week3OpenPrice','Week3ClosePrice','Week3LowPrice','Week3HighPrice','Week3Volume','Week3Amount','Week3Diff','Week3Percent','Week3Exchange','Week4OpenPrice','Week4ClosePrice','Week4LowPrice','Week4HighPrice','Week4Volume','Week4Amount','Week4Diff','Week4Percent','Week4Exchange']    
predictNames = ['NextDayHighPriceIncrease','NextDayHighPriceIncrease1','NextDayHighPriceIncrease2','NextDayHighPriceIncrease3','NextDayHighPriceIncrease4','NextDayHighPriceIncrease5','NextDayHighPriceIncrease6','NextDayHighPriceIncrease7','NextDayHighPriceIncrease8','NextDayHighPriceIncrease9','NextDayHighPriceIncrease10','NextDayHighPriceDecrease1','NextDayHighPriceDecrease2','NextDayHighPriceDecrease3','NextDayHighPriceDecrease4','NextDayHighPriceDecrease5','NextDayHighPriceDecrease6','NextDayHighPriceDecrease7','NextDayHighPriceDecrease8','NextDayHighPriceDecrease9','NextDayHighPriceDecrease10','NextDayClosePriceIncrease','NextDayClosePriceIncrease1','NextDayClosePriceIncrease2','NextDayClosePriceIncrease3','NextDayClosePriceIncrease4','NextDayClosePriceIncrease5','NextDayClosePriceIncrease6','NextDayClosePriceIncrease7','NextDayClosePriceIncrease8','NextDayClosePriceIncrease9','NextDayClosePriceIncrease10','NextDayClosePriceDecrease1','NextDayClosePriceDecrease2','NextDayClosePriceDecrease3','NextDayClosePriceDecrease4','NextDayClosePriceDecrease5','NextDayClosePriceDecrease6','NextDayClosePriceDecrease7','NextDayClosePriceDecrease8','NextDayClosePriceDecrease9','NextDayClosePriceDecrease10','Next3DayHighPriceIncrease','Next3DayHighPriceIncrease3','Next3DayHighPriceIncrease5','Next3DayHighPriceIncrease7','Next3DayHighPriceIncrease9','Next3DayHighPriceIncrease11','Next3DayHighPriceIncrease13','Next3DayHighPriceIncrease15','Next3DayHighPriceIncrease17','Next3DayHighPriceIncrease20','Next3DayHighPriceIncrease25','Next3DayHighPriceDecrease3','Next3DayHighPriceDecrease5','Next3DayHighPriceDecrease7','Next3DayHighPriceDecrease9','Next3DayHighPriceDecrease11','Next3DayHighPriceDecrease13','Next3DayHighPriceDecrease15','Next3DayHighPriceDecrease17','Next3DayHighPriceDecrease20','Next3DayHighPriceDecrease25','Next3DayClosePriceIncrease','Next3DayClosePriceIncrease3','Next3DayClosePriceIncrease5','Next3DayClosePriceIncrease7','Next3DayClosePriceIncrease9','Next3DayClosePriceIncrease11','Next3DayClosePriceIncrease13','Next3DayClosePriceIncrease15','Next3DayClosePriceIncrease17','Next3DayClosePriceIncrease20','Next3DayClosePriceIncrease25','Next3DayClosePriceDecrease3','Next3DayClosePriceDecrease5','Next3DayClosePriceDecrease7','Next3DayClosePriceDecrease9','Next3DayClosePriceDecrease11','Next3DayClosePriceDecrease13','Next3DayClosePriceDecrease15','Next3DayClosePriceDecrease17','Next3DayClosePriceDecrease20','Next3DayClosePriceDecrease25','Next5DayHighPriceIncrease','Next5DayHighPriceIncrease3','Next5DayHighPriceIncrease5','Next5DayHighPriceIncrease7','Next5DayHighPriceIncrease9','Next5DayHighPriceIncrease11','Next5DayHighPriceIncrease13','Next5DayHighPriceIncrease15','Next5DayHighPriceIncrease17','Next5DayHighPriceIncrease20','Next5DayHighPriceIncrease25','Next5DayHighPriceDecrease3','Next5DayHighPriceDecrease5','Next5DayHighPriceDecrease7','Next5DayHighPriceDecrease9','Next5DayHighPriceDecrease11','Next5DayHighPriceDecrease13','Next5DayHighPriceDecrease15','Next5DayHighPriceDecrease17','Next5DayHighPriceDecrease20','Next5DayHighPriceDecrease25','Next5DayClosePriceIncrease','Next5DayClosePriceIncrease3','Next5DayClosePriceIncrease5','Next5DayClosePriceIncrease7','Next5DayClosePriceIncrease9','Next5DayClosePriceIncrease11','Next5DayClosePriceIncrease13','Next5DayClosePriceIncrease15','Next5DayClosePriceIncrease17','Next5DayClosePriceIncrease20','Next5DayClosePriceIncrease25','Next5DayClosePriceDecrease3','Next5DayClosePriceDecrease5','Next5DayClosePriceDecrease7','Next5DayClosePriceDecrease9','Next5DayClosePriceDecrease11','Next5DayClosePriceDecrease13','Next5DayClosePriceDecrease15','Next5DayClosePriceDecrease17','Next5DayClosePriceDecrease20','Next5DayClosePriceDecrease25']

df = pd.read_sql("SELECT {} from {} where  DATABALANCE = -1".format(getSQLColumnString(featureNames + predictNames),tableName), 
                 conn, 
                 columns=featureNames + predictNames)



X_train = df[featureNames][:].as_matrix()
y_test = df[predictCategory][:].as_matrix()


X_resampled, y_resampled = SMOTE(ratio={1:60000},kind='borderline1').fit_sample(X_train, y_test)

newdf = pd.DataFrame(data=X_resampled,columns=featureNames)
newdf[predictCategory] = y_resampled


tableName = 'SZ50Bank5D3C'
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
    
for index, row in newdf.iterrows():
    insertAExample(row,cursor,conn)



