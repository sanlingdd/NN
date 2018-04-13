import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from datetime import datetime
import traceback  
import os
import json
import requests
import traceback

def tryconvert(value):
    try:
        return np.float64(value)
    except ValueError:
        return 0
    return 0

def getSingleStockDaily(code,start,end):
    site = 'http://q.stock.sohu.com/hisHq?code=cn_{}&start={}&end={}&stat=1&order=D&period=d&rt=json';
    site = site.format(code,start,end)
    response =requests.get(site)
    
    html = response.text
    htmljson = json.loads(html)
    try:
        data = htmljson[0].get('hq')
    except Exception:
        print(code)
        return
    
    df = pd.DataFrame(data=data, columns=['Date', 'OpenPrice','ClosePrice','Diff','Percent', 'LowPrice', 'HighPrice',  'Volume', 'Amount','Exchange'])
    df['Percent'] = df['Percent'].apply(lambda x : x[:len(x) - 1])
    df['Exchange'] = df['Exchange'].apply(lambda x : x[:len(x) - 1])
    df['Percent'] = df['Percent'].apply(lambda x :  tryconvert(x))
    df['Exchange'] = df['Exchange'].apply(lambda x : tryconvert(x))
    df.fillna(value=0)
    
    try:
        df.astype (dtype={'Date':str,'OpenPrice':np.float64,'ClosePrice':np.float64,'Diff':np.float64,'Percent':np.float64,'LowPrice':np.float64,'HighPrice':np.float64,  'Volume':np.float64, 'Amount':np.float64,'Exchange':np.float64})
    except ValueError:
        print(sys.exc_info())
    
    

    
    return df
def priceAdjustion(df):
    ratio = 1
    for index, row in df.iterrows():
        currentRowNum = df.index.get_loc(index)
        if currentRowNum > 0:
            rowPrevious = df.iloc[currentRowNum - 1]

            if ratio != 1:
                df.set_value(index, 'OpenPrice', row['OpenPrice'] * ratio) 
                df.set_value(index, 'HighPrice', row['HighPrice'] * ratio) 
                df.set_value(index, 'LowPrice', row['LowPrice'] * ratio) 
                df.set_value(index, 'ClosePrice', row['ClosePrice'] * ratio)
                df.set_value(index, 'Volume', row['Volume'] * ratio)   
                df.set_value(index, 'Diff', row['Diff'] * ratio)  
                row = df.iloc[currentRowNum]
                            
            if  row['ClosePrice'] - rowPrevious['ClosePrice'] / (1 + rowPrevious['Percent']) > 0.01:
                ratio = ratio * (rowPrevious['ClosePrice'] / (1 + rowPrevious['Percent'])) / row['ClosePrice'] 
                df.set_value(index, 'OpenPrice', row['OpenPrice'] * ratio) 
                df.set_value(index, 'HighPrice', row['HighPrice'] * ratio) 
                df.set_value(index, 'LowPrice', row['LowPrice'] * ratio) 
                df.set_value(index, 'ClosePrice', row['ClosePrice'] * ratio)
                df.set_value(index, 'Volume', row['Volume'] * ratio)   
                df.set_value(index, 'Diff', row['Diff'] * ratio)  
               

    return df
    


if __name__ == '__main__':
    code = '002343'
    df = pd.read_csv('data/{}'.format(code), index_col='Date', parse_dates=True,na_values=['nan'])
    df['Percent'] = df['Percent'].apply(lambda x : (x / 100))
    df['Exchange'] = df['Exchange'].apply(lambda x : (x / 100))

    dfAdjust = priceAdjustion(df)
#read the stock daily price and stock into text
#     codes= ['zs_000001','zs_000016','002343']
#     for code in codes:
#         start = '19890101'
#         end = '20171020'
#         df = getSingleStockDaily(code,start,end)
#         filename = 'data/{}'.format(code);
#         if os.path.exists(filename):
#             df.to_csv(filename,index=False,mode='a', header=None)
#         else:
#             df.to_csv(filename,index=False)


