
# coding: utf-8

# In[1]:


import os
path = 'D:\my documents\Courses\Capstone Project\Data'
os.chdir(path)
import Realized_Volatility_and_Term_Volatility as vol
import calendar
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
# n = 5 
# start_date = '2017-01-01'
# end_date = '2018-01-01'
# y = [2017, 2018, 2019, 2020]
# m = [3, 6, 9, 12]
# tau = 0.5589
# name = 'U8'
# rw = [5, 10, 21, 63]

    
def IntrplFurRate(tau, df):
    if (tau > 1/4) & (tau <= 2/4):
        y1, y2 = df.iloc[:,0], df.iloc[:,1]
        x1, x2 = 1/4, 2/4
    elif (tau > 2/4) & (tau <= 3/4):
        y1, y2 = df.iloc[:,1], df.iloc[:,2]
        x1, x2 = 2/4, 3/4
    elif (tau > 3/4) & (tau <= 4/4):
        y1, y2 = df.iloc[:,2], df.iloc[:,3]
        x1, x2 = 3/4, 4/4
    k = (y1 -y2)/(x1 - x2)
    c = (y2*x1 - y1*x2)/(x1 - x2)
    #print(k[1], c[1])
    return k*tau + c

    
def GetPCvol(df, name):
    real_vol_1w, real_vol_2w, real_vol_1m, real_vol_3m = vol.CallFunc(vol.Realized_Vol, df, rw)
    real_vol = [real_vol_1w, real_vol_2w, real_vol_1m, real_vol_3m]

    # col = ['1w', '2w', '1m', '3m']

    real_vol_CM = vol.CM_realized_vol(real_vol)

    CM_PC = vol.Vol_PC(real_vol_CM, name)
    return CM_PC


def Est_vol_para(CMT, tau, name, rw):

    
    # In[2]:
       
    #print("Using CMT ret to interpolate for Sep.18 option,", " Time to maturity is " + str(tau))
 
    U8CMT = IntrplFurRate(tau, CMT)
  
    # In[3]:
        
    U8CMT_ret = U8CMT.diff()/U8CMT
    
    vt = GetPCvol(U8CMT_ret, name)**2 # v_t is the variance
    dvt = vt.diff().dropna()
    dRt_Rt = U8CMT_ret[-dvt.shape[0]:] # it's dRt/Rt
    
    cov = np.cov(dvt, dRt_Rt)
    
    lbd, R = np.linalg.eig(cov)

    rho = cov[0][1] / np.sqrt(cov[0][0]) /np.sqrt(cov[1][1])
    
    
    # In[7]:
    
    
    u2t = R[1][1]*dvt
    var_u2t = GetPCvol(u2t, name)**2
    var_u2t
    
    
    # In[8]:   
    
    # Create linear regression object
    regr = linear_model.LinearRegression()
    
    # Train the model using the training sets
    res = regr.fit(vt[-var_u2t.shape[0]:].reshape(-1, 1), var_u2t.reshape(-1, 1))
    sigma_square = res.coef_[0][0]
    #print('res.coef_', res.coef_)
    sigma = np.sqrt(sigma_square)
  
    return  rho, sigma


if __name__ == "__main__":
    ###########################################################################
    n = 5 # Settle prices of rst 8 (most liquid) rolling Eurodollar futures you got from Quandl
    start_date = '2017-01-01'
    end_date = '2018-01-01'
    
    _, in_sample = vol.DownloadData(n, start_date, end_date)
    
    y = [2017, 2018, 2019, 2020]
    m = [3, 6, 9, 12]   
    c = calendar.Calendar(firstweekday = calendar.SUNDAY)
      
    df_CMT = vol.CMT_rate(in_sample, y, m, c)
    CMT = df_CMT[[ 'CM_ED' + str(i) for i in range(1, n)]] #[[str(i * 3) + 'm_days' for i in range(5, 21)] + [ 'CM_ED' + str(i) for i in range(5, 21)]]
    ###########################################################################

    ###########################################################################
    import Discount_Factor_by_Interpolated_LIBOR_rate as DF
    TTM, _ = DF.TTM()   
    tau_list = np.array(TTM)/365 # tau = 0.5589

    ###########################################################################
    name_list = ['M8', 'U8', 'Z8'] # name
#    rw_list = [[5, 10, 21, 63], [5, 15, 42, 63], [10, 21, 42, 63], [15, 21, 42, 63],
#               np.random.randint(low = 5, high=63, size=4)] #rw = [5, 10, 21, 63]
    
    def Gen_rw(low, high, size):
        return np.random.randint(low, high, size=4)
    
    rw_list = [Gen_rw(5, 63, 4) for i in range(10)]
#    rho_array = np.array(())
#    rho, sigma = Est_vol_para(CMT, tau, name, rw)
#    print("rho, sigma are: ", rho, sigma)
    
###############################
    print ("%10s %10s %10s %10s %10s %10s %20s" % ('M8_rho', 'U8_rho', 'Z8_rho', 'M8_sigma', 'U8_sigma', 'Z8_sigma', 'rolling_window'))
    print ("="*70)
    
    rho1_list = np.full((10,), 0.0)
    rho2_list = np.full((10,), 0.0)
    rho3_list = np.full((10,), 0.0)
    sigma1_list = np.full((10,), 0.0)
    sigma2_list = np.full((10,), 0.0)
    sigma3_list = np.full((10,), 0.0)
    vol_para_list  = np.full((10,6), 0.0)
    avg = 0
    for i, rw in enumerate(rw_list):
        rho1, sigma1 = Est_vol_para(CMT, tau_list[0], name_list[0], rw)
        rho2, sigma2 = Est_vol_para(CMT, tau_list[1], name_list[1], rw)
        rho3, sigma3 = Est_vol_para(CMT, tau_list[2], name_list[2], rw)
        vol_para_list[i,:] = rho1, rho2, rho3, sigma1, sigma2, sigma3
        print ("%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f  %20s" % (rho1, rho2, rho3, sigma1, sigma2, sigma3, rw))

    print ("-"*70)
    #print ("Average Abs Error (%%) : %5.3f" % (avg))
    
###  Strikes Quantilib Value  My Model Value   Relative Error (%)
    x = np.array(range(1, 11))
    plt.figure(figsize = (10, 10))
    plt.plot(x, vol_para_list)
    plt.legend(['M8_rho', 'U8_rho', 'Z8_rho', 'M8_sigma', 'U8_sigma', 'Z8_sigma'], fontsize = 10)
    plt.show()