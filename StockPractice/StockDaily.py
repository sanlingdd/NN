# coding=utf-8

import pandas as pd
import sys
import os
import numpy as np

date = '2006/01/04'
stockNumberFile = "SH600321.csv"
location = r'C:\Users\i071944\Documents\Stk_1F_2006\Stk_1F_2006'

def getDailyTradeInfo(date, stockNumberFile):
    df = pd.read_csv(location + "\\" + stockNumberFile,header=None,
                     names=['Date','Time','OpenPrice','LowPrice','HighPrice','ClosePrice','Volume','Amount'],
                     dtype={'Date':str,'Time':str,'LowPrice':np.float64,'HighPrice':np.float64,'ClosePrice':np.float64,'Volume':np.float64,'Amount':np.float64
                            }
                     )
    dfd = df[df['Date'] == date]
    OpenPrice = dfd.head(1)['OpenPrice']
    ClosePrice = dfd.tail(1)['ClosePrice']
    LowPrice = dfd['LowPrice'].min()
    HighPrice = dfd['HighPrice'].max()
    Volume = dfd.groupby('Date')['Volume'].sum()
    Amount = dfd.groupby('Date')['Amount'].sum()
    return pd.DataFrame(data = [[date], [OpenPrice],[ClosePrice],[LowPrice], [HighPrice], [Volume], [Amount]])

def getTradeInfoInRange(from_date, to_date,stockNumberFile):
    df = pd.read_csv(location + "\\" + stockNumberFile,header=None,
                     names=['Date','Time','OpenPrice','LowPrice','HighPrice','ClosePrice','Volume','Amount'],
                     dtype={'Date':str,'Time':str,'LowPrice':np.float64,'HighPrice':np.float64,'ClosePrice':np.float64,'Volume':np.float64,'Amount':np.float64
                            }
                     )
    dfd = df[df['Date'].isin([from_date,to_date])]
    OpenPrice = dfd.head(1)['OpenPrice']
    ClosePrice = dfd.tail(1)['ClosePrice']    
    LowPrice = dfd['LowPrice'].min()
    HighPrice = dfd['HighPrice'].max()
    Volume = dfd.groupby('Date')['Volume'].sum()
    Amount = dfd.groupby('Date')['Amount'].sum()
    return pd.DataFrame(data = [[date], [OpenPrice],[ClosePrice],[LowPrice], [HighPrice], [Volume], [Amount]])

df = pd.read_csv(location + "\\" + stockNumberFile,header=None,
                 names=['Date','Time','OpenPrice','LowPrice','HighPrice','ClosePrice','Volume','Amount'],
                 dtype={'Date':str,'Time':str,'LowPrice':np.float64,'HighPrice':np.float64,'ClosePrice':np.float64,'Volume':np.float64,'Amount':np.float64
                        }
                 )

sff = df.groupby(['Date']).apply(lambda x: x.mean())
trade_dates = list(sff.index).sort();


print(getDailyTradeInfo(date,stockNumberFile))


