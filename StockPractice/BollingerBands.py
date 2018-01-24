"""Bollinger Bands."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo


def compute_cumulative_returns(df):
    """Compute and return the cumulative return values."""
    daily_return = compute_daily_returns(df)
    return daily_return.cumsum()


def compute_daily_returns(df):
    """Compute and return the daily return values."""
    # TODO: Your code here
    # Note: Returned DataFrame must have the same number of rows
    df_return = (df / df.shift(1)) - 1
    #df_return.ix[0,:] = 0
    df_return.fillna(0,inplace=True)
    return df_return
#     df_return = df.copy()
#     df_return[1:] = (df[1:] / df[:-1].values) - 1
#     df_return.ix[0,:] = 0
#     return df_return

def fill_missing_values(df_data):
    """Fill missing values in data frame, in place."""
    ##########################################################
    df_data.fillna(method='ffill',inplace=True)
    df_data.fillna(method='bfill',inplace=True)
    pass  # TODO: Your code here (DO NOT modify anything else)
    ##########################################################

def symbol_to_path(symbol, base_dir="data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}".format(str(symbol)))


def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if '000016' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, '000016')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='date',
                parse_dates=True, usecols=['date', 'close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'close': symbol})
        df = df.join(df_temp,how='inner')
        #if symbol == 'SPY':  # drop dates SPY did not trade
        #    df = df.dropna(subset=["SPY"])

    return df


def plot_data(df, title="Stock prices"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()


def get_rolling_mean(values, window):
    """Return rolling mean of given values, using specified window size."""
    return values.rolling(window,center=False).mean()
    #pd.rolling_mean(values, window=window)


def get_rolling_std(values, window):
    """Return rolling standard deviation of given values, using specified window size."""
    # TODO: Compute and return rolling standard deviation
    return values.rolling(window,center=False).std()
    #pd.rolling_std(values, window=window)


def get_bollinger_bands(rm, rstd):
    """Return upper and lower Bollinger Bands."""
    # TODO: Compute upper_band and lower_band
    upper_band = rm + 2*rstd
    lower_band = rm - 2*rstd
    
    return upper_band, lower_band


def test_run():
    # Read data
    dates = pd.date_range('2015-01-01', '2016-12-31')
    
    symbols = ['000016']
    df = get_data(symbols, dates)
    windowSize = 20
    # Compute Bollinger Bands
    # 1. Compute rolling mean
    rm_SPY = get_rolling_mean(df['000016'], window=windowSize).ix[windowSize:]

    # 2. Compute rolling standard deviation
    rstd_SPY = get_rolling_std(df['000016'], window=windowSize).ix[windowSize:]

    df = df.ix[windowSize:]
    # 3. Compute upper and lower bands
    upper_band, lower_band = get_bollinger_bands(rm_SPY, rstd_SPY)
    
    # Plot raw SPY values, rolling mean and Bollinger Bands
    ax = df['000016'].plot(title="Bollinger Bands", label='000016')
    rm_SPY.plot(label='Rolling mean', ax=ax)
    upper_band.plot(label='upper band', ax=ax)
    lower_band.plot(label='lower band', ax=ax)
    # Add axis labels and legend
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper right')
    plt.show()
    
        
    pricePassLow = (df['000016'] < lower_band) 
    
    pricePassUpper = (df['000016'] > upper_band)
    
    buyed = False
    cost = 0.
    first = True
    profit = 0.
    for index,(buy,sell) in enumerate(zip(pricePassLow,pricePassUpper)):
        if not buyed and buy:
            buyed = True
            cost = df.ix[index]
        if sell and not first:
            profit += df.ix[index] - cost
            buyed=False
            first = True
    

    
     
    daily_return = compute_daily_returns(df)
    
    #scatterplot SPY and GLD
    daily_return.plot(kind='scatter',x='000016',y='600606')
    w, b = np.polyfit(daily_return['000016'],daily_return['600606'],1)
    plt.plot(daily_return['000016'],w*daily_return['000016'] + b,'-',color='r')
    plt.show()
    
    #correlation
    df_cor = daily_return.corr(method='pearson')
    print(df_cor)

    print ('a')
    

def getPortforlioSharpRatio(dates,symbols,allocate):
    # Read data
    #dates = pd.date_range('2016-01-01', '2016-12-31')
    #symbols = ['000016','600848','600606']
    df = get_data(symbols, dates)
    #daily_return = compute_daily_returns(df)
    df_normal = df / df.ix[0]
    #allocate = pd.Series({'000016': 0.3, '600848': 0.3, '600606': 0.4})
    startValue = 1
    df_alocate = df_normal[:] * allocate
    pos_val = df_alocate * startValue
    daily_portforlia_val = pos_val.sum(axis=1)
    daily_return = compute_daily_returns(daily_portforlia_val)
    
    risk_free_rate_return = 0#.05 # or 0.05(5%) national debt return rate
    if daily_return.std() == 0:
       return 0
    sharpratio = 15.87 * (daily_return.mean() -risk_free_rate_return) / daily_return.std()
    #print(sharpratio)
    return sharpratio

def getPortforlioCumulativeReturn(dates,symbols,allocate):
    df = get_data(symbols, dates)
    #cumulative return
    df_normal = df.ix[-1] - df.ix[0]
    #
    startValue = 1
    df_alocate = df_normal[:] * allocate
    pos_val = df_alocate * startValue
    cumulative_return = pos_val.sum()
    return cumulative_return
    
    
def test_runSharpRatio():
    dates = pd.date_range('2016-01-01', '2016-12-31')
    symbols = ['000016','600848','600606']
    allocate = pd.Series({'000016': 0.3, '600848': 0.3, '600606': 0.4})
    sharpratio = getPortforlioSharpRatio(dates,symbols,allocate)   
    print(sharpratio) 
    
    symbols = ['002466']
    allocate = pd.Series({'002466': 1})
    sharpratio = getPortforlioSharpRatio(dates,symbols,allocate)    
    print(sharpratio) 

def PortforlioSharpRatioFunction(allocate,dates,symbols):
    allocPDSer = pd.Series()
    for (alloc,symbol) in zip(allocate,symbols):
        allocPDSer=allocPDSer.append(pd.Series({symbol:alloc}))
    
    sharpratio = getPortforlioSharpRatio(dates,symbols,allocPDSer)
    return sharpratio * (-1)

def PortforlioCumulativeReturnFunction(allocate,dates,symbols):
    allocPDSer = pd.Series()
    for (alloc,symbol) in zip(allocate,symbols):
        allocPDSer=allocPDSer.append(pd.Series({symbol:alloc}))
    
    sharpratio = getPortforlioCumulativeReturn(dates,symbols,allocPDSer)
    return sharpratio * (-1)

def cons(allocs):
    sum = 0.
    for alloc in allocs:
        sum+=alloc
    
    sum = 1 - sum
    return sum
 
def opimizePortforlio():
    #To get the largest sharp ratio
    dates = pd.date_range('2016-01-01', '2016-12-31')
    symbols = ['000016','600848','600606']
    #df = get_data(symbols, dates)

    bonds = []
    for x in range(len(symbols)):
        bonds.append((0,1))


    allocate =[] #np.random.normal(0,0.1,len(symbols))
    percent = 1 / len(symbols)
    for x in range(len(symbols)):
        allocate.append(percent)
    print(PortforlioCumulativeReturnFunction([0,0,1],dates,symbols))
    #cons = ({'type': 'eq', 'fun': lambda x:  x[0] + x[1] + x[2] - 1})
    print(PortforlioCumulativeReturnFunction([0,1,0],dates,symbols))
    print(PortforlioCumulativeReturnFunction([1,0,0],dates,symbols))
    
    result = spo.minimize(PortforlioCumulativeReturnFunction,allocate,args=(dates,symbols),method='SLSQP',
                          bounds=bonds,
                          #constraints=({'type': 'eq', 'fun': lambda x: cons(x)}),
                          options={'disp':True}
                          )
    print(PortforlioCumulativeReturnFunction(result.x,dates,symbols)) 
    print(result.x)
    print(result.x)
    
if __name__ == "__main__":
    test_run()
    #test_runSharpRatio()
    opimizePortforlio()
