
# coding: utf-8

# #### Compute a discount factor, df(D,option expiry) using interpolated LIBOR rate
# 
# Fix at 2/15/2018

# In[1]:


import os

path = 'D:\my documents\Courses\Capstone Project\Data'
os.chdir(path)
os.getcwd()


# In[2]:


import pandas as pd
USD_LIBOR_rates = pd.read_excel("USD_LIBOR_rates_2_23_2018.xlsx")


# In[3]:


USD_LIBOR_rates.index = USD_LIBOR_rates['USD']
LIBOR_rates = USD_LIBOR_rates['2/23/2018']
LIBOR_rates = LIBOR_rates[LIBOR_rates != '-'] # remove nan
LIBOR_rates = LIBOR_rates.str.slice(0, 7) # keep first 6 elements in the string
LIBOR_rates = LIBOR_rates.convert_objects(convert_numeric=True) # convert string into float
LIBOR_rate = LIBOR_rates[4:].values
LIBOR_dict = {'ov': LIBOR_rates[0], '1w': LIBOR_rates[1], '1m': LIBOR_rates[2], '2m': LIBOR_rates[3],               '3m': LIBOR_rates[4], '6m': LIBOR_rates[5], '12m': LIBOR_rates[6] }


# In[4]:


# set calculation_date
from datetime import date
calculation_date = date(2018, 2, 25)


# set settlement_date
import calendar
c = calendar.Calendar(firstweekday = calendar.SUNDAY) # Sunday as the first day of the week

y = [2018]
m = [6, 9, 12]

monthcal = [c.monthdatescalendar(year,month) for year in y for month in m]# + [c.monthdatescalendar(2007,3)]
# separate months into weeks

mondays = [[day for week in month for day in week if                 day.weekday() == calendar.MONDAY and                 day.month in m] for month in monthcal ]

settlement_date = [mondays[i][2] for i in range(len(mondays))]


# In[6]:


import numpy as np
from dateutil.relativedelta import relativedelta
taus = np.array(settlement_date) - calculation_date
# map(timedelta.days, taus) # 'member_descriptor' object is not callable

# will use first 3m libor, and the left will use interpolate
# =============================================================================
# columns = ['Time_to_maturity', 'Annulized_TTM', 'Days_need_to_interpolate']
# Days = pd.DataFrame(index = [1, 2, 3], columns = columns)
# =============================================================================

def TTM(): 
    Time_to_maturity = [None] * len(taus)
    LIBOR_days = [None] * len(taus)
    ############################ for this project only
    LIBOR_month = [3, 6, 12]
    ############################ for this project only
    for i in range(len(taus)):
        Time_to_maturity[i] = taus[i].days
        LIBOR_days[i] = ( (calculation_date + relativedelta(months = LIBOR_month[i])) - calculation_date).days

    return Time_to_maturity, LIBOR_days
# =============================================================================
# Days.loc[:, 'Time_to_maturity'] = Time_to_maturity
# Days.loc[:, 'Annulized_TTM'] = Days['Time_to_maturity'] / 365 
# Days.loc[:, 'LIBOR_days'] = LIBOR_days
# Days.loc[:, 'LIBOR_rate'] = LIBOR_rates[4:].values
# =============================================================================



# In[7]:


def Interpolate_rate(LIBOR_rate1, LIBOR_rate2, LIBOR_days1, LIBOR_days2, TTM_days):
    slope = (LIBOR_rate1 - LIBOR_rate2) / (LIBOR_days1 - LIBOR_days2)
    #print(slope)
    intercept = LIBOR_rate1 - LIBOR_days1 * slope
    #print(intercept)
    predict = intercept + slope * TTM_days
    print(predict)
    return 1 / (1 + predict * TTM_days / 365 / 100) 

def Disc_fac():   
    Time_to_maturity, LIBOR_days = TTM()
    DF = [None] * len(taus) # Discount factor
    DF[0] = Interpolate_rate(LIBOR_rate[0], LIBOR_rate[1], LIBOR_days[0], LIBOR_days[1], Time_to_maturity[0])
    DF[1] = Interpolate_rate(LIBOR_rate[1], LIBOR_rate[2], LIBOR_days[1], LIBOR_days[2], Time_to_maturity[1])
    DF[2] = Interpolate_rate(LIBOR_rate[1], LIBOR_rate[2], LIBOR_days[1], LIBOR_days[2], Time_to_maturity[2])
    return DF

