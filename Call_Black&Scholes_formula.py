#Valuation of European Call Option with Black and Scholes formula. 

import numpy as np
from scipy.stats import norm

def formula_BS(S0, K, T, r, sig):
  
# d1 
  d1 = (np.log(S0 / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))

# d2
  d2 = d1 - sig * np.sqrt(T)

# Call option price(C)
  # norm.cdf(x) Normal standard cumulative distribution function of X 
     
  # norm.cdf(d1) --> Call Option Delta  
    
  # norm.cdf(d2) --> probabily of the Option expiring "in the money" (ITM)
   
  C = S0 * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)
  
  return C
