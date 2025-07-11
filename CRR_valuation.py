#European/American Call/Put Option with binomial method 
#Cox-Ross-Rubinstein (CRR)

import numpy as np

def CRR(S0, r, sigma, T, K, N, call=True, eur=True):
    
    
    dt = T / N # Time step size
    nu = r - 0.5 * sigma**2 # Drift (average trend) of the asset price
    fs = np.exp(-r * dt) # Discount factor per step

    u = np.exp(sigma * np.sqrt(dt))    # Up factor
    d = np.exp(-sigma * np.sqrt(dt))    # Down factor
    q = 0.5 + 0.5 * (nu * np.sqrt(dt)) / sigma    # Risk-neutral probability of an upward movement

    # Stock prices at maturity
    ST = S0 * (d ** np.arange(N, -1, -1)) * (u ** np.arange(0, N + 1))

    # Payoff of a call option at maturity 
    if call:
        H = np.maximum(ST - K, 0)
    else:
        H = np.maximum(K - ST, 0)

    
    # Initialize option value matrix
    D = np.zeros((N + 1, N + 1))    
    D[:, -1] = H    

    # Backward induction
    for i in range(N - 1, -1, -1): # Iterate backwards from maturity to present
        for j in range(i + 1): # Iterate through each node at the current time step
            # Calculate option value at the current node using risk-neutral probabilities and discount factor
            D[j, i] = fs * (q * D[j, i + 1] + (1 - q) * D[j + 1, i + 1])

            # For American options, check for early exercise
            if not eur:
                # Calculate the stock price at the current node (S_t,i)
                ST_ij = S0 * (d ** (N - i + j)) * (u ** (i - j))
                # Calculate the intrinsic value (IV) if exercised early at this node
                if call:
                    IV = max(ST_ij - K, 0)
                else:
                    IV = max(K - ST_ij, 0)
                # Take the maximum of holding the option or exercising early
                D[j, i] = max(D[j, i], IV)

    return D[0, 0] # Return the option price at time 0 (the first element of the matrix)
