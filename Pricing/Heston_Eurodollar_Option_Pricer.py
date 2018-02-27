
# coding: utf-8

# ##### 2F Heston Model:
# This is to price European option of LIBOR future, under T+delta forward measure(where LIBOR is martingale)

# In[158]:


import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

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
#         self.k = k
#         self.theta = theta
        self.sigma = sigma
        self.rho = rho
        
        self.mu1 = 1/2
        self.mu2 = -1/2
        self.a = k * theta
        self.b1 = k + lambda_ - rho * sigma
        self.b2 = k + lambda_
        
        self.option_type = option_type        
        self.i = 1j
        
    def d1(self, phi):
        d1 = np.sqrt((self.rho*self.sigma*self.i*phi - self.b1)**2 - self.sigma**2*(2*self.mu1*self.i*phi - phi**2))
        return d1
    def d2(self, phi):
        d2 = np.sqrt((self.rho*self.sigma*self.i*phi - self.b2)**2 - self.sigma**2*(2*self.mu2*self.i*phi - phi**2))
        return d2
    def g1(self, phi):
        g1 = (self.b1-self.rho*self.sigma*self.i*phi+self.d1(phi)) / (self.b1-self.rho*self.sigma*self.i*phi-self.d1(phi))
        return g1
    def g2(self, phi):
        g2 = (self.b2-self.rho*self.sigma*self.i*phi+self.d2(phi)) / (self.b2-self.rho*self.sigma*self.i*phi-self.d2(phi))
        return g2
    
    def D(self, phi, choice):
        if choice == 1:
            D = (self.b1 - self.rho*self.sigma*self.i*phi + self.d1(phi)) / self.sigma**2         *((1 - np.exp(self.d1(phi)*self.tau)) / (1 - self.g1(phi)*np.exp(self.d1(phi)*self.tau)))
            if np.isnan(D):
                D = (self.b1 - self.rho*self.sigma*self.i*phi + self.d1(phi)) / self.sigma**2 / self.g1(phi)
        else:
            D = (self.b2 - self.rho*self.sigma*self.i*phi + self.d2(phi)) / self.sigma**2         *((1 - np.exp(self.d2(phi)*self.tau)) / (1 - self.g2(phi)*np.exp(self.d2(phi)*self.tau)))
            if np.isnan(D):
                 D = (self.b2 - self.rho*self.sigma*self.i*phi + self.d2(phi)) / self.sigma**2 / self.g1(phi)
        return D

    def C(self, phi, choice):
        if choice == 1:
            C = self.r0*self.i*phi*self.tau + self.a/self.sigma**2 *         ((self.b1-self.rho*self.sigma*self.i*phi+self.d1(phi))*self.tau -          2*np.log((1-self.g1(phi)*np.exp(self.d1(phi)*self.tau))/(1-self.g1(phi))))
            if np.isnan(C):
                C = self.r0*self.i*phi*self.tau + self.a/self.sigma**2 *         ((self.b1-self.rho*self.sigma*self.i*phi+self.d1(phi))*self.tau -          2*(self.d1(phi)*self.tau))
        
        else:
            C = self.r0*self.i*phi*self.tau + self.a/self.sigma**2 *         ((self.b2-self.rho*self.sigma*self.i*phi+self.d2(phi))*self.tau -          2*np.log((1-self.g2(phi)*np.exp(self.d2(phi)*self.tau))/(1-self.g2(phi))))
            if np.isnan(C):
                C = self.r0*self.i*phi*self.tau + self.a/self.sigma**2 *         ((self.b2-self.rho*self.sigma*self.i*phi+self.d2(phi))*self.tau -          2*(self.d1(phi)*self.tau))

        return C

    def f(self, phi, choice): # characteristic function
        ######## x = ln(S0)
        return np.exp(self.C(phi, choice) + self.D(phi, choice)*self.v0 + self.i*phi*np.log(self.S0))
    
    def Integrand(self, phi, choice):
        return np.real(np.exp(-self.i*phi*np.log(self.K))*self.f(phi, choice)/(self.i*phi))
        
    def P(self, choice): # probility
        integral = quad(self.Integrand, 0, np.inf, (choice))
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

import warnings
warnings.filterwarnings("ignore")



def Price_diff(sigma, P_mkt, S0, K, tau, option_type):
    if option_type == "C":
        m = HestonModel(0.0, 0.04, S0, K, tau, 0.0, 0.1, 0.04, sigma, -0.75, 0.0, "C")
    else:
        m = HestonModel(0.0, 0.04, S0, K, tau, 0.0, 0.1, 0.04, sigma, -0.75, 0.0, "P")
    return abs(P_mkt - m.Price())
    
def LN_IV(P_mkt, S0, K, tau, option_type):
    result = minimize(Price_diff, 0.1, args = (P_mkt, S0, K, tau, option_type), bounds = ((0, 1),)) # fun, x0, args 
    return result.x





