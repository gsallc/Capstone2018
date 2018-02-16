#2F Heston Model:
#dr(t)=mu*dt+vol*dW(t)

import numpy as np
from scipy.integrate import quad

# under T+delta forward measure, LIBOR is martingale
class Model:
    def __init__(self, r0, S0, K, tau, BondP, mu, k, theta, sigma, rho, lambda_):
        ##### phi, i
        self.r0 = 0
        self.S0 = S0
        self.K = K
        self.tau = tau
        self.BondP = BondP # P(t,T) bond price
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
        
        self.i = 1j
    def d1(self, phi):
        return np.sqrt((self.rho*self.sigma*self.i*phi - self.b1)**2 - self.sigma**2*(2*self.mu1*self.i*phi - phi**2))
    def d2(self, phi):
        return np.sqrt((self.rho*self.sigma*self.i*phi - self.b2)**2 - self.sigma**2*(2*self.mu2*self.i*phi - phi**2))
    def g1(self, phi):
        return (self.b1-self.rho*self.sigma*self.i*phi+self.d1(phi)) / (self.b1-self.rho*self.sigma*self.i*phi-self.d1(phi))
    def g2(self, phi):
        return (self.b2-self.rho*self.sigma*self.i*phi+self.d2(phi)) / (self.b2-self.rho*self.sigma*self.i*phi-self.d2(phi))
    
    def D(self, phi, choice):
        if choice == 1:
            return (self.b1 - self.rho*self.sigma*self.i*phi + self.d1(phi)) / self.sigma**2 \
        *((1 - np.exp(self.d1(phi)*self.tau)) / (1 - self.g1(phi)*np.exp(self.d1(phi)*self.tau)))
        else:
            return (self.b2 - self.rho*self.sigma*self.i*phi + self.d2(phi)) / self.sigma**2 \
        *((1 - np.exp(self.d2(phi)*self.tau)) / (1 - self.g2(phi)*np.exp(self.d2(phi)*self.tau)))

    def C(self, phi, choice):
        if choice == 1:
            return self.r0*self.i*phi*self.tau + self.a/self.sigma**2 * \
        ((self.b1-self.rho*self.sigma*self.i*phi+self.d1(phi))*self.tau - \
         2*np.log((1-self.g1(phi)*np.exp(self.d1(phi)*self.tau))/(1-self.g1(phi))))
        else:
            return self.r0*self.i*phi*self.tau + self.a/self.sigma**2 * \
        ((self.b2-self.rho*self.sigma*self.i*phi+self.d2(phi))*self.tau - \
         2*np.log((1-self.g2(phi)*np.exp(self.d2(phi)*self.tau))/(1-self.g2(phi))))

    def f(self, phi, choice): # characteristic function
        ######## x = ln(S0)
        return np.exp(self.C(phi, choice) + self.D(phi, choice) + self.i*phi*np.log(self.S0))
    
    def Integrand(self, phi, choice):
        return np.real(np.exp(-self.i*phi*np.log(self.K))*self.f(phi, choice)/(self.i*phi))
        
    def P(self, choice):
        integral = quad(self.Integrand, 0, np.inf, (choice))
        print(choice)
        print(integral)
        return 1/2 + 1/np.pi * (integral[0] - integral[1])
    
    def CallPrice(self):
        return self.S0*self.P(1) - self.K*self.BondP*self.P(2)
    
    
# =============================================================================
# # (self, r0, S0, K, tau, BondP, mu, k, theta, sigma, rho, lambda_)
# m1 = Model(0, 100, 100, 1, 0.98, 0.1, 0.2, 0.1, 0.2, 0.3, .2)
# m1.CallPrice()
# =============================================================================
