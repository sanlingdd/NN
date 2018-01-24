import sys
sys.path.append('..')

import pandas as pd
import pymysql
from pymongo import MongoClient
import json
import tushare as ts
from tushare.stock import cons as cs
from io import StringIO
import urllib
from datetime import datetime
import traceback  


# proxyurl = 'http://www.xicidaili.com/nn/{}'.format('1')
# proxy_handler = urllib.request.ProxyHandler({'http': 'http://proxy.pal.sap.corp:8080/'})
# opener = urllib.request.build_opener(proxy_handler)
# lines = opener.open(proxyurl,timeout=10).read()
# lines = lines.decode('UTF-8') 
# 
# print(datetime.now())

#proxy='http://api.xicidaili.com/free2016.txt'
#proxy_handler = urllib.request.ProxyHandler({'http': 'http://proxy.sha.sap.corp:8080/'})
#opener = urllib.request.build_opener(proxy_handler)
#lines = opener.open(proxy,timeout=10).read()
#lines = lines.decode('GBK') 


def get_tick_data_history_sina(date,code,proxy,port,pause):
    symbol = cs._code_to_symbol(code)
    url = 'http://market.finance.sina.com.cn/downxls.php?date={}&symbol={}'.format(date,symbol)
    proxy_handler = urllib.request.ProxyHandler({'https': 'https://{}:{}/'.format(proxy,port)})
    opener = urllib.request.build_opener(proxy_handler)
    lines = opener.open(url,timeout=pause).read()
    lines = lines.decode('GBK') 
    if len(lines) < 20:
        return None
    df = pd.read_table(StringIO(lines), names=cs.TICK_COLUMNS,
                       skiprows=[0]) 
    return df

print(datetime.now())


# proxiesList = [('proxy.pvgl.sap.corp',8080),('proxy.sha.sap.corp',8080),('proxy.pek.sap.corp',8080),('proxy.hkg.sap.corp',8080),('proxy.sin.sap.corp',8080),('proxy.syd.sap.corp',8080),('proxy.tyo.sap.corp',8080),('proxy.wdf.sap.corp',8080),('proxy.pal.sap.corp',8080),('proxy.phl.sap.corp',8080), ('proxy.osa.sap.corp',8080)] 
# 
# for proxy in proxiesList:
#     proxyhttp,port = proxy
#     try:
#         df = get_tick_data_history_sina('2017-01-10','600606',proxyhttp,port,pause=10)
#     except IOError as e:
#         traceback.print_exc()  
#     except ValueError:
#         traceback.print_exc()  
#     except:
#         traceback.print_exc()  
        
proxiesList = [('proxy.pvgl.sap.corp',8080),('proxy.sha.sap.corp',8080),('proxy.pek.sap.corp',8080),('proxy.hkg.sap.corp',8080),('proxy.sin.sap.corp',8080),('proxy.syd.sap.corp',8080),('proxy.tyo.sap.corp',8080),('proxy.wdf.sap.corp',8080),('proxy.pal.sap.corp',8080),('proxy.phl.sap.corp',8080), ('proxy.osa.sap.corp',8080)] 

try:
    df = get_tick_data_history_sina('2017-01-10','000016','125.112.175.23','38503',pause=10)
except IOError as e:
    traceback.print_exc()  
except ValueError:
    traceback.print_exc()  
except:
    traceback.print_exc()          
    
    
print(datetime.now())


client = MongoClient('localhost', 27017)
db = client.stock

db.TradeDays.insert(json.loads(df.to_json(orient='records')))

