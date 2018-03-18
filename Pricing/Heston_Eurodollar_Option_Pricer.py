
# coding: utf-8

# ##### 2F Heston Model:
# This is to price European option of LIBOR future, under T+delta forward measure(where LIBOR is martingale)

import numpy as np
from scipy.integrate import quad

class HestonModel:
    def __init__(self, r0, v0, S0, K, tau, mu, k, theta, sigma, rho, lambda_, option_type):
        ##### phi, i
        self.r0 = r0
        self.v0 = v0
        self.S0 = S0
        self.K = K
        self.tau = tau
#         self.BondP = BondP # P(t,T) bond price
        ##### maybe it is not necessary
#         self.mu = mu
        self.k = k
        self.lambda_ = lambda_
#         self.theta = theta
        self.sigma = sigma
        self.rho = rho
        
        self.mu1 = 1/2
        self.mu2 = -1/2
        self.a = k * theta
#         self.b1 = k + lambda_ - rho * sigma
#         self.b2 = k + lambda_
        
        self.option_type = option_type        
        self.i = 1j
    
    def b(self, choice):
        if choice == 1:
            b = self.k + self.lambda_ - self.rho * self.sigma
        else:
            b = self.k + self.lambda_
        return b    
        
    def d(self, phi, choice):
        if choice == 1:
            d = np.sqrt((self.rho*self.sigma*self.i*phi - self.b(choice))**2 - self.sigma**2*(2*self.mu1*self.i*phi - phi**2))
        else:
            d = np.sqrt((self.rho*self.sigma*self.i*phi - self.b(choice))**2 - self.sigma**2*(2*self.mu2*self.i*phi - phi**2))
        return d
    
    def g(self, phi, choice):
        if choice == 1:
            g = (self.b(choice)-self.rho*self.sigma*self.i*phi+self.d(phi, choice)) / (self.b(choice)-self.rho*self.sigma*self.i*phi-self.d(phi, choice))
        else:
            g = (self.b(choice)-self.rho*self.sigma*self.i*phi+self.d(phi, choice)) / (self.b(choice)-self.rho*self.sigma*self.i*phi-self.d(phi, choice))
        return g
       
    def D(self, phi, choice):
        ######## here is the problem: b1
        D = (self.b(choice) - self.rho*self.sigma*self.i*phi + self.d(phi, choice)) / self.sigma**2 \
        *((1 - np.exp(self.d(phi, choice)*self.tau)) / (1 - self.g(phi, choice)*np.exp(self.d(phi, choice)*self.tau)))
        return D

    def C(self, phi, choice):
        C = self.r0*self.i*phi*self.tau + self.a/self.sigma**2 * ((self.b(choice)-self.rho*self.sigma*self.i*phi+self.d(phi, choice))*self.tau - \
         2*np.log((1-self.g(phi, choice)*np.exp(self.d(phi, choice)*self.tau))/(1-self.g(phi, choice))))
        return C

    def f(self, phi, choice): # characteristic function
        ######## x = ln(S0)
        return np.exp(self.C(phi, choice) + self.D(phi, choice)*self.v0 + self.i*phi*np.log(self.S0))
    
    def Integrand(self, phi, choice):
        #print(phi)
        return np.real(np.exp(-self.i*phi*np.log(self.K))*self.f(phi, choice)/(self.i*phi))
        
    def P(self, choice):
        integral = quad(self.Integrand, 0, np.inf, (choice))#, points = [500])
        return 1/2 + 1/np.pi * (integral[0] - integral[1])
    
    def Price(self):
        if self.option_type == "C":
            return (self.S0*self.P(1) - self.K*self.P(2))
        else:
            return ( self.K*(1 - self.P(2)) - self.S0*(1 - self.P(1)) )
    # Under RN, this hould be (self.S0*self.P(1) - BondP*self.K*self.P(2))
    
    

# In[165]:


# S0 = 127.62, K =130, tau = 1, BondP = 1, v0 = 0.04; kappa = 0.1; theta = 0.04; rho = -0.75; sigma = 0.1;
# (self, r0, v0, S0, K, tau, mu, k, theta, sigma, rho, lambda_, option_type)

# =============================================================================
# import warnings
# warnings.filterwarnings("ignore")
# =============================================================================





# =============================================================================
# def Price_diff(v0, P_mkt, S0, K, tau, option_type):
#     if option_type == "C":
#         m = HestonModel(0.0, v0, S0, K, tau, 0.0, 0.1, 0.04, 0.1, -0.75, 0.0, "C")
#     else:
#         m = HestonModel(0.0, v0, S0, K, tau, 0.0, 0.1, 0.04, 0.1, -0.75, 0.0, "P")
#     return abs(P_mkt - m.Price())
#     
# def LN_IV(P_mkt, S0, K, tau, option_type):
#     result = minimize(Price_diff, 0.4, args = (P_mkt, S0, K, tau, option_type), \
#                       method = 'SLSQP', bounds = ((0.01, 1),)) # fun, x0, args 
#     return result.x
# =============================================================================





