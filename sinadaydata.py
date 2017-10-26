# coding=utf-8

import urllib
import zlib
import json
import requests
import pandas as pd
import sys
import numpy as np
import pymysql


site = 'http://q.stock.sohu.com/hisHq?code=cn_%s&start=%s&end=%s&stat=1&order=D&period=d&rt=json';

response =requests.get(site)

html = response.text
gzipped = response.headers.get('Content-Encoding')#查看是否服务器是否支持gzip
if gzipped:
    html = zlib.decompress(html, 16+zlib.MAX_WBITS)#解压缩，得到网页源码
htmljson = json.loads(html)
data = htmljson[0].get('hq')

df = pd.DataFrame(data=data, columns=['Date', 'OpenPrice','ClosePrice','Diff','Percent', 'LowPrice', 'HighPrice',  'Volume', 'Amount','Exchange']
                 )
df['Percent'] = df['Percent'].apply(lambda x : x[:len(x) - 1])
df['Exchange'] = df['Exchange'].apply(lambda x : x[:len(x) - 1])
df.astype (dtype={'Date':str,'OpenPrice':np.float64,'ClosePrice':np.float64,'Diff':np.float64,'Percent':np.float64,'LowPrice':np.float64,'HighPrice':np.float64,  'Volume':np.float64, 'Amount':np.float64,'Exchange':np.float64})

df


