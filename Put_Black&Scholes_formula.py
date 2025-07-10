import numpy as np
from scipy.stats import norm

def BS_formula(S0, K, T, r, sig):
  # d1 
  d1 = (np.log(S0 / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))

  # d2
  d2 = d1 - sig * np.sqrt(T)
  
#Put Option Price (P) 

  P = K * np.exp(-r*T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

return P
  
