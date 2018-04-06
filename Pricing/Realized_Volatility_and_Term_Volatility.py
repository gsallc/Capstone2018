
# coding: utf-8

# #### Realized_Volatility

# In[1]:


import quandl
import pandas as pd
token = "m3kA-wQ4xZPzt5Khy7xA"



# In[2]:


# Convert to constant maturity
import calendar
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

c = calendar.Calendar(firstweekday = calendar.SUNDAY)

y = [2017, 2018, 2019, 2020]
m = [3, 6, 9, 12]

def CMT_rate(df, y, m):
    df_test = df.copy()
    print('Calculate settlement day')
    n = df_test.shape[1]
    monthcal = [c.monthdatescalendar(year,month) for year in y for month in m]# + [c.monthdatescalendar(2007,3)]

    third_wednesday = [[day for week in month for day in week if                     day.weekday() == calendar.WEDNESDAY and                     day.month in m] for month in monthcal ]

    settlement_day = [third_wednesday[i][2] for i in range(len(third_wednesday))]
    print('Calculate time to maturity, Actual expiration days')
    for index, row in df_test.iterrows():

        tau_list = np.array(settlement_day) - index.date() 
        tau_list = tau_list[tau_list >= timedelta(days = 2)][:n] 

        for i in range(n):
            df_test.loc[index, 'Tau' + str(i + 1)] = tau_list[i].days # time to maturity
            df_test.loc[index, str((i + 1) * 3) + 'm_days'] = ((index.date() + relativedelta(months = 3 * (i+ 1))) - index.date()).days # 3m contract days
    
    print('Interpolate rate to constant maturity')
    for index, row in df_test.iterrows():
        for i in range(n - 1):
#             if row['Tau' + str(i + 1)] > row[str((i + 1) * 3) + 'm_days']:
#                 df_test.loc[index, 'CM_ED' + str(i + 1)] = row['Settle_ED' + str(i + 1)] / row['Tau' + str(i + 1)] * row[str((i + 1) * 3) + 'm_days']
#             else:
            x = [row['Tau' + str(i + 1)], row['Tau' + str(i + 2)]]
            y = [row['Settle_ED' + str(i + 1)], row['Settle_ED' + str(i + 2)]]

            coefficients = np.polyfit(x, y, 1)
            p = np.poly1d(coefficients)

            df_test.loc[index, 'CM_ED' + str(i + 1)] = p(row[str((i + 1) * 3) + 'm_days'])
    
    return df_test



# In[3]:


# 1w, 2w, 1m, 3m

def Realized_Vol(df, rolling_windodw):
    vol = df.rolling(window = rolling_windodw).std() * np.sqrt(252) # annualize it
    return vol


# In[ ]:
def CallFunc(func, para1, para2):
    a = func(para1, para2[0])
    b = func(para1, para2[1])
    c = func(para1, para2[2])
    d = func(para1, para2[3])
    return a, b, c, d

# In[4]:


import matplotlib.pyplot as plt


def Plot(df, numrow, numcol, numseries, rolling_length = ''):
    n = df.shape[1]
    for i in range(n):     
        plt.subplot(numrow, numcol, i * numcol + numseries)
        plt.plot(range(len(df.index)), df.iloc[:, i])
        plt.title('Realized Volatility ' + str(df.columns[i]) + '  ' + rolling_length + '  ', fontsize = 15)
        plt.ylabel('Volatility', fontsize = 12)
        
        
        a = np.array([np.array([i , x.month]) for i, x in enumerate(df.index) ])
        b = np.unique(a[:,1], return_index = True)
        xlocs = b[1][0:len(b[1]):3]
        xlabels = df.index[xlocs].strftime("%Y-%m")
        #print(xlocs, xlabels)
        plt.xticks(xlocs, xlabels, fontsize = 15) # , rotation = 15
        plt.yticks(fontsize = 15)



# ### using PCA1 as vol proxy/factor

# In[5]:

col = ['1w', '2w', '1m', '3m']

def CM_realized_vol(data, column):
    df = pd.concat([x[column] for x in real_vol], join = 'inner', axis = 1)
    df.columns = col
    df.dropna(inplace = True, axis = 0)
    return df


# In[6]:


from sklearn.decomposition import PCA

def Vol_PC(data, column):
    #data = scale(data)
#     pca = PCA(n_components=1)
#     pca.fit(data)
#     X1=pca.fit_transform(data)

    normalized_data = (data - data.mean(axis = 0))
    pca = PCA(n_components = 1)
    pca.fit(normalized_data)
    print(pca.components_)
    X1 = (pca.components_ * normalized_data).sum(axis = 1) + data.mean(axis = 1).mean()
    X1.name = column
    return X1

#     mean = data.mean(axis = 0)
#     std = data.std(axis = 0)
#     print(data.head())
#     normalized_data = (data - mean)/std

#     pca = PCA(n_components = 1)
#     pca.fit(normalized_data)
#     print('Variance explained: ', pca.explained_variance_ratio_)  

#     df = (pca.components_ * normalized_data)
#     df = df * std + mean
#     df = df.sum(axis = 1)

#     df.name = column
#     return df

# In[7]:


# from sklearn.decomposition import PCA

# def Vol_PC(data, column):
#     mean = data.mean(axis = 0)
#     std = data.std(axis = 0)
#     print(data.head())
#     normalized_data = (data - mean)/std

#     pca = PCA(n_components = 1)
#     pca.fit(normalized_data)
#     print('Variance explained: ', pca.explained_variance_ratio_)  

#     df = (pca.components_ * normalized_data)
#     print(df.head())
#     print(mean, std)
#     df = df * std + mean
#     print(df.head())
#     df = df.sum(axis = 1)

#     df.name = column
#     return df


# CM_ED1_PC = Vol_PC(real_vol_CM_ED1, 'CM_ED1')
# CM_ED2_PC = Vol_PC(real_vol_CM_ED2, 'CM_ED2')
# CM_ED3_PC = Vol_PC(real_vol_CM_ED3, 'CM_ED3')
# CM_ED4_PC = Vol_PC(real_vol_CM_ED4, 'CM_ED4')

# CM_ED_PC = pd.concat([CM_ED1_PC, CM_ED2_PC, CM_ED3_PC, CM_ED4_PC], axis = 1)
# print(CM_ED_PC.head())

# real_vol_CM_ED1.mean().mean(), real_vol_CM_ED2.mean().mean(), real_vol_CM_ED3.mean().mean(), real_vol_CM_ED4.mean().mean(), CM_ED_PC.mean()


# In[8]:



# ### term volatility (also known as integrated volatility) as sigma_term(T) = sqrt[1/T*integral{s=0,T}(sigma(s)^2*ds]
# $$ \sigma^{term}(T) = \sqrt{\frac{1}{T} \int\limits_{0}^{T} \sigma(s)^2 ds } $$

# In[9]:


from scipy import interpolate

def cubic_spline(x, x_points, y_points):

    tck = interpolate.splrep(x_points, y_points)
    return interpolate.splev(x, tck)

#print (cubic_spline(np.array(range(1,10))))

# In[29]:


def Sigma_Term(term, df):   # CM_ED_PC
    n_seq = np.array(range(int(term*365)))/365
    sigma_term = np.full((len(n_seq), ), 0.0)

#     temp = CM_ED_PC.iloc[-1, :].values
#     temp = np.insert(temp, 0, temp[0])
    sigma_s = cubic_spline(n_seq, np.array([91, 182, 273, 365])/365, CM_ED_PC.iloc[-1, :])
    print(df.iloc[0, :])
    ###########
#     from scipy.interpolate import interp1d

#     f = interp1d(np.array([90, 180, 270, 360])/252, df.iloc[0, :], kind='cubic')

#     plt.plot(np.array([90, 180, 270, 360])/252, df.iloc[0, :], 'o', n_seq[90:], f(n_seq[90:]), '.-', markersize = 3)
#     plt.legend(['data', 'cubic'], loc='best')
#     plt.show()
    ##############
    for i in range(len(n_seq)):
        sigma_term[i] = np.sqrt(sum(sigma_s[:i] ** 2) / 365 / n_seq[i])

    print("Term vol for Sep.18,", "sigma_term(T): T is ranging from 0 to 0.5589")
    plt.plot(np.array([91, 182, 273, 365])/365, CM_ED_PC.iloc[-1, :], 'o', )
    plt.plot(n_seq, sigma_s)
    plt.plot(n_seq, sigma_term)
    plt.legend(['data', 'sigma_s', 'sigma_term'])
    plt.title('Term Volatility')
    plt.xticks( rotation = 30 )
    plt.show()

    return sigma_term


#########################################################################
    
print('Downloading ED from Quandl')

n = 5 # Settle prices of rst 8 (most liquid) rolling Eurodollar futures you got from Quandl
nms = ['CHRIS/CME_ED' + str(i) for i in range(1, n + 1)]
dfs = [quandl.get(nm, authtoken = token, start_date="2017-01-01") for nm in nms] #, end_date="2018-03-07"
dfs = [data.Settle for data in dfs]
df = pd.concat(dfs, axis = 1)
df.columns = ['Settle_ED' + str(i) for i in range(1, n + 1)]
df = 100 - df # convert into rate
in_sample = df[df.index.year == 2017]
#in_sample.head()

print('Construct constant maturity rate')

df_CMT = CMT_rate(in_sample, y, m)
CMT = df_CMT[[ 'CM_ED' + str(i) for i in range(1, n)]] #[[str(i * 3) + 'm_days' for i in range(5, 21)] + [ 'CM_ED' + str(i) for i in range(5, 21)]]
#CMT.head()
#df_CMT.to_csv(r'C:\Users\passi\OneDrive\Desktop\df_CMT.csv')
#df_CMT.head(60)

# use the diff, see the Heston, we are interested in the lognormal vol, not the normal vol.
CMT_ret = CMT.diff()

real_vol_1w, real_vol_2w, real_vol_1m, real_vol_3m = CallFunc(Realized_Vol, CMT_ret, [5, 10, 21, 63])
real_vol = [real_vol_1w, real_vol_2w, real_vol_1m, real_vol_3m]

# =============================================================================
# fig = plt.figure(1)
# plt.figure(figsize = (30, 40))
# 
# col = ['1w', '2w', '1m', '3m']
# 
# Plot(real_vol_1w, 8, 4, 1, col[0])    
# Plot(real_vol_2w, 8, 4, 2, col[1]) 
# Plot(real_vol_1m, 8, 4, 3, col[2]) 
# Plot(real_vol_3m, 8, 4, 4, col[3]) 
# plt.show()
# 
# real_vol_1w.mean()
# 
# =============================================================================


real_vol_CM_ED1, real_vol_CM_ED2, real_vol_CM_ED3, real_vol_CM_ED4 = CallFunc(CM_realized_vol, real_vol, ['CM_ED1', 'CM_ED2', 'CM_ED3', 'CM_ED4'])


CM_ED1_PC = Vol_PC(real_vol_CM_ED1, 'CM_ED1')
CM_ED2_PC = Vol_PC(real_vol_CM_ED2, 'CM_ED2')
CM_ED3_PC = Vol_PC(real_vol_CM_ED3, 'CM_ED3')
CM_ED4_PC = Vol_PC(real_vol_CM_ED4, 'CM_ED4')

CM_ED_PC = pd.concat([CM_ED1_PC, CM_ED2_PC, CM_ED3_PC, CM_ED4_PC], axis = 1)
#print(CM_ED_PC.head())

#real_vol_CM_ED1.mean().mean(), real_vol_CM_ED2.mean().mean(), real_vol_CM_ED3.mean().mean(), real_vol_CM_ED4.mean().mean(), CM_ED_PC.mean()

# =============================================================================
# 
# fig = plt.figure(1)
# plt.figure(figsize = (20, 20))
# 
# colnames = ['CM_ED1', 'CM_ED2', 'CM_ED3', 'CM_ED4']
# Plot(CM_ED_PC, 4, 1, 1)    
# 
# plt.show()
# 
# =============================================================================


#term_vol = Sigma_Term(0.8, real_vol_1w.iloc[:,0])
# np.sqrt( (real_vol_1w.iloc[:, 0] ** 2).mean()  * 251/252)
print("Term vol for Sep.18,", "sigma_term(T): T is fixed for 0.5589")
term_vol = Sigma_Term(0.5589, CM_ED_PC)
plt.plot(term_vol)
plt.title('Term Volatility')
plt.xticks( rotation = 30 )
plt.show()


a = Sigma_Term(1, CM_ED_PC)



