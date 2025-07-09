import numpy as np
from scipy.stats import norm

def formula_BS(S0, K, T, r, sig):
  
# d1 
  d1 = (np.log(S0 / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))

# d2
  d2 = d1 - sig * np.sqrt(T)

# Call option price(C)
  # norm.cdf(x) Normal standard cumulative distribution function of X 
     
  # norm.cdf(d1) --> Delta (sensibilità) dell'opzione Call.  
    
  # norm.cdf(d2) --> probabilità (sotto la misura neutrale al rischio) che l'opzione scada "in the money" (ITM)
   
  C = S0 * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)
  
  return C
