
# coding: utf-8

# In[1]:


import os

path = 'D:\my documents\Courses\Capstone Project\Data'
os.chdir(path)
os.getcwd()


# In[2]:


import pandas as pd

def Get_option_data(file_name):
    df = pd.read_excel(file_name)
    df = df.loc[df.index[2]:df.index[len(df)-2],['Strike', 'Type', 'Settle', 'Estimated Volume', 'Prior Day Open Interest']]
    # df.index[len(df)-2] will be included
    df = df[df.Settle != 'CAB']
    df.reset_index(inplace = True)
    return df
    
M8 = Get_option_data("ED_Option_M8_2_23_2018.xlsx")
U8 = Get_option_data("ED_Option_U8_2_23_2018.xlsx")
Z8 = Get_option_data("ED_Option_Z8_2_23_2018.xlsx")


# In[3]:


# Get current ED future price
ED_Future = pd.read_excel("ED_Futures_Settlements_2_23_2018.xlsx")
ED_Future.set_index('Month', inplace = True)
F0 = ED_Future.loc[['JUN 18', 'SEP 18', 'DEC 18'], 'Settle']


# In[4]:


import Discount_Factor_by_Interpolated_LIBOR_rate as DF
Disc_fac = DF.Disc_fac()
TTM, _ = DF.TTM()


# In[5]:


# def Black_term(price, disc_fac):
#     return price/disc_fac


# In[17]:


import numpy as np
import BS_Option_Implied_Vol as BS

def Get_LN_IV(sigma_ini, df, Type_, S0, tau, disc_fac):
    # print(Type_, S0, tau, disc_fac)
    data = df[df.Type == Type_]
    data.reset_index(inplace = True)
    length = data.shape[0]
    IV = np.full((1, length), .0).reshape(-1,) # make sure that all your scores have the same type.
    # scoreA = np.array([float(1) / (i + 1) for i in range(len(a))],dtype=float)
    
    if Type_ == "Call":
        for i in range(length):    
            BS_m = BS.BSModel(S0, data.loc[i, 'Strike'] / 100, tau, "C")
            sigma_est = BS_m.find_vol(data.loc[i, 'Settle'] / disc_fac)
            IV[i] = sigma_est#[0]

    else:
        for i in range(length):    
            #IV[i] = BS.LN_IV(sigma_ini, data.loc[i, 'Settle'] / disc_fac, S0, data.loc[i, 'Strike'] / 100, tau, "P")
            BS_m = BS.BSModel(S0, data.loc[i, 'Strike'] / 100, tau, "P")
            sigma_est = BS_m.find_vol(data.loc[i, 'Settle'] / disc_fac)
            IV[i] = sigma_est
    return IV


# In[19]:

import matplotlib.pyplot as plt
print("Calculating implied volatility for Calls:")
M8_IV_C = Get_LN_IV(0.4, M8, "Call", F0[0], TTM[0]/365, Disc_fac[0])

M8_IV_P = Get_LN_IV(0.04, M8, "Put", F0[0], TTM[0]/365, Disc_fac[0])

U8_IV_C = Get_LN_IV(0.04, U8, "Call", F0[1], TTM[1]/365, Disc_fac[1])

U8_IV_P = Get_LN_IV(0.04, U8, "Put", F0[1], TTM[1]/365, Disc_fac[1])

Z8_IV_C = Get_LN_IV(0.04, Z8, "Call", F0[2], TTM[2]/365, Disc_fac[2])

Z8_IV_P = Get_LN_IV(0.04, Z8, "Put", F0[2], TTM[2]/365, Disc_fac[2])



# In[27]:


plt.figure(1)
plt.figure(figsize=(20,10))
plt.subplot(231)
plt.plot(M8.loc[M8.Type == "Call", 'Strike'], M8_IV_C)
plt.title("M8 Call")

plt.subplot(234)
plt.plot(M8.loc[M8.Type == "Put", 'Strike'], M8_IV_P)
plt.title("M8 Put")

plt.subplot(232)
plt.plot(U8.loc[U8.Type == "Call", 'Strike'], U8_IV_C)
plt.title("U8 Call")

plt.subplot(235)
plt.plot(U8.loc[U8.Type == "Put", 'Strike'], U8_IV_P)
plt.title("U8 Put")

plt.subplot(233)
plt.plot(Z8.loc[Z8.Type == "Call", 'Strike'], Z8_IV_C)
plt.title("Z8 Call")

plt.subplot(236)
plt.plot(Z8.loc[Z8.Type == "Put", 'Strike'], Z8_IV_P)
plt.title("Z8 Put")
plt.show()


# In[12]:


# a = np.linspace(0.25, .26, num=11)
# for i in range(len(a)):
#     print(a[i],"aaa")
#     b =BS.Price_diff(a[i], 6.55, 127.62, 130, 0.31, 'C')
#     print(b)


# In[13]:


# def Get_LN_IV(df, Type_, S0, tau, disc_fac):
#     print(Type_, S0, tau, disc_fac)
#     data = df[df.Type == Type_]
#     length = data.shape[0]
#     IV = np.full((1, length), 1).reshape(-1,)
#     if Type_ == "Call":
#         option_type = "C"
#     else:
#         option_type = "P" 
    
#     for i in range(length):   
#         m = BS.BSModel(S0, data.loc[i, 'Strike'] / 100, tau, option_type)
#         print("i")
#         print(i)
#         print(data.loc[i, 'Settle'] / disc_fac)
#         sigma_est = m.find_vol(data.loc[i, 'Settle'] / disc_fac)
#         print("sigma_est")
#         print(sigma_est)
#         IV[i] = sigma_est
#         print("IV[i]")
#         print(IV[i])

#     return IV


# In[14]:


# Get_LN_IV(M8, "Call", F0[0], TTM[0]/365, Disc_fac[0])


# In[15]:


# m = BS.BSModel(97.76, 91.25, 0.3095890410958904, "C")
# a = np.linspace(0.0, .4, num=11)
# for i in range(len(a)):
#     print(m.Price(a[i]))


# In[16]:


M8.loc[M8.Type == "Call", :]


# In[23]:


Disc_fac

