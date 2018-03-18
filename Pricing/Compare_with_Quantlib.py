
# coding: utf-8

# #### Compare with Quantlib

# In[10]:


import QuantLib as ql

v0 = 0.04; k = kappa = 0.1; theta = 0.04; rho = -0.75; sigma = 0.1; S0 = spot = 97.76; K = 91.25; tau = 0.31; BondP = 1; mu = 0; Type = -1;
exercise = ql.EuropeanExercise(ql.Date(18,6,2018)) # (maturity_date)
calculation_date = ql.Date(23,2,2018)

def Pricing_by_QuantLib(v0, S0, K, tau, mu, kappa, theta, sigma, rho, Type):
    # construct the European Option
    payoff = ql.PlainVanillaPayoff(Type, K) # (option_type, strike_price) enum Type { Put = -1,. Call = 1. }   
    european_option = ql.VanillaOption(payoff, exercise)    
    
    
    ql.Settings.instance().evaluationDate = calculation_date
    day_count = ql.Actual365Fixed()

    #dividend_yield = ql.QuoteHandle(ql.SimpleQuote(0.0))
    risk_free_rate = 0.0
    dividend_rate = 0.0

    flat_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, risk_free_rate, day_count))

    dividend_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, dividend_rate, day_count))
    
    
    # parameters initialization
    heston_process = ql.HestonProcess(flat_ts, dividend_ts, 
                               ql.QuoteHandle(ql.SimpleQuote(spot)), 
                               v0, kappa, theta, sigma, rho)
    
    
    engine = ql.AnalyticHestonEngine(ql.HestonModel(heston_process)) #,0.01, 1000)
    european_option.setPricingEngine(engine)
    h_price = european_option.NPV()
#    print ("The Heston model price is", h_price)
#     ql.HestonModel(2.3) # HestonProcessPtr const &
#     iv = european_option.impliedVolatility(6.51, heston_process)
#     # VanillaOptionPtr::impliedVolatility(Real,GeneralizedBlackScholesProcessPtr const &)
#     print("iv is ", iv)
    return h_price


# In[11]:


# import os
# path = 'D:\my documents\Courses\Capstone Project\Data'
# os.chdir(path)
import Heston_Eurodollar_Option_Pricer as HM

Pricing_by_QuantLib(0.3, S0, K, tau, mu, kappa, theta, sigma, rho, Type)
m1 = HM.HestonModel(0.0, 0.3, S0, K, tau, mu, k, theta, sigma, rho, 0.0, "P")
print('My model price is ', m1.Price())


# In[12]:


import numpy as np
strikes = np.linspace(91, 100, 20)

print ("%15s %15s %15s %20s" % (
    "Strikes", "Quantilib Value", 
     "My Model Value", "Relative Error (%)"))
print ("="*70)

avg = 0
for i, opt in enumerate(strikes):
    q = Pricing_by_QuantLib(0.3, S0, opt, tau, mu, kappa, theta, sigma, rho, Type)
    m = HM.HestonModel(0.0, 0.3, S0, opt, tau, mu, k, theta, sigma, rho, 0.0, "P").Price()
    err = (m/q - 1.0)
    print ("%15.2f %14.5f %15.5f %20.7f " % (strikes[i], q, m, 100.0*(m/q - 1.0)))
    avg += abs(err)
avg = avg*100.0/len(strikes)
print ("-"*70)
print ("Average Abs Error (%%) : %5.3f" % (avg))


# In[9]:


## tau is 1yr, but quantlib gives different answer

