import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo

def f(X):
    Y = (X - 1.5)**2 + 0.5
    print( "X = {}, Y = {}".format(X,Y))
    return Y

def error(line, data):
    err = np.sum((data[:,1] - (line[0]*data[:,0] + line[1]))**2)
    return err

def fit_line(data,err):
    #initial guess
    line = np.float32([0,np.mean(data[:,1])])
    
    #plot initial guess
    x_ends = np.float32([-5,5])
    plt.plot(x_ends,line[0]*x_ends + line[1],'m--',linewidth=2,label='Initial guess')
    
    result = spo.minimize(err,line,args=(data,),method='SLSQP',options={'disp':True})
    return result.x

def test_run():
    l_org = np.float32([4,2])
    print("Original line: C0={}, C1={}".format(l_org[0],l_org[1]))
    xorg = np.linspace(0,10,21)
    yorg = l_org[0]*xorg + l_org[1]
    plt.plot(xorg,yorg,'b--',linewidth=2,label='Original Line')
    
    #noise data
    noise_sigma = 3
    noise = np.random.normal(0,noise_sigma,yorg.shape)
    
    data = np.asarray([xorg,yorg+noise]).T
    plt.plot(data[:,0],data[:,1],'go',label='Data Point')
    
    l_fit = fit_line(data,error)
    print("fit line C0:{}, C1:{}".format(l_fit[0],l_fit[1]))
    plt.plot(data[:,0],l_fit[0] * data[:,0] + l_fit[1],'r--',linewidth=2,label='fitted line')
    
    plt.legend(loc='upper left')
    plt.show()
    print('a')
    
if __name__ == "__main__":
    test_run()
    
    
    
