
# coding: utf-8

# ##### 2F Heston Model:
# This is to price European option of LIBOR future, under T+delta forward measure(where LIBOR is martingale)

# In[158]:


import numpy as np
from scipy.optimize import minimize
import scipy.stats as ss

class BSModel:
    #Black and Scholes
    def __init__(self, S0, K, tau, option_type):
        self.S0 = S0
        self.K = K
        self.r = 0.0
        self.tau = tau
        #self.sigma = sigma
        self.option_type = option_type
    
    def d1(self, sigma):
        return (np.log(self.S0 / self.K) + (self.r + sigma**2 / 2) * self.tau) / (sigma * np.sqrt(self.tau))

    def d2(self, sigma):
        return (np.log(self.S0 / self.K) + (self.r - sigma**2 / 2) * self.tau) / (sigma * np.sqrt(self.tau))

    def Price(self, sigma):
        if self.option_type == "C":
            return self.S0 * ss.norm.cdf(self.d1(sigma)) - self.K * np.exp(-self.r * self.tau) * ss.norm.cdf(self.d2(sigma))
        else:
            return self.K * np.exp(-self.r * self.tau) * ss.norm.cdf(-self.d2(sigma)) - self.S0 * ss.norm.cdf(-self.d1(sigma))

    def Vega(self, sigma):
        return self.S0 * np.sqrt(self.tau)*self.d1(sigma)

    def find_vol(self, P_mkt): # bisection-method, as price is monotonically increasing in volatility
        PRECISION = 1.0e-5
    
        sigma_left, sigma_right= 0.0, 0.5
        while sigma_right - sigma_left > PRECISION:
            sigma_mid = (sigma_left + sigma_right) / 2
            if self.Price(sigma_mid) - P_mkt > 0:
                sigma_right = sigma_mid
            else:
                sigma_left = sigma_mid

        return sigma_mid


    
# =============================================================================
#     def find_vol(self, P_mkt):
#         MAX_ITERATIONS = 100
#         PRECISION = 1.0e-5
#     
#         sigma = 0.5
#         for i in range(0, MAX_ITERATIONS):
#             price = self.Price(sigma)
#             vega = self.Vega(sigma)
#             
#             price = price
#             diff = P_mkt - price  # our root
# # =============================================================================
# #             print(price, diff, sigma, vega)
# #             print(Price_diff(sigma, P_mkt, self.S0, self.K, self.tau, self.option_type))
# # =============================================================================
#             #print (i, sigma, diff)
#     
#             if (abs(diff) < PRECISION):
#                 return sigma
#             sigma = sigma + diff/vega # f(x) / f'(x)
#     
#         # value wasn't found, return best guess so far
#         return sigma
# =============================================================================



def Price_diff(sigma_est, P_mkt, S0, K, tau, option_type):
    if option_type == "C":
        BS_m = BSModel(S0, K, tau, "C")
    else:
        BS_m = BSModel(S0, K, tau, "P")
# =============================================================================
#     print(BS_m.Price(sigma_est))
# =============================================================================
    return abs(P_mkt - BS_m.Price(sigma_est)) #abs(BS_m.Price() - HM_m.Price())
     
 
def LN_IV(sigma_ini, P_mkt, S0, K, tau, option_type):
    result = minimize(Price_diff, sigma_ini, args = (P_mkt, S0, K, tau, option_type), \
                      method = 'CG')#, tol=1e-6, bounds = ((0.0, 1),)) # fun, x0, args 
# =============================================================================
#     print(result.x)
#     print("Price_diff")
#     print(Price_diff(result.x, P_mkt, S0, K, tau, option_type))
# =============================================================================
    return result.x


