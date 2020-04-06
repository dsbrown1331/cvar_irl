import numpy as np
import matplotlib.pyplot as plt

def brute_force_solve_cvar(vals, probs, alpha):
    max_cvar = -np.float('inf')
    sigma_star = None
    all_sigma = []
    all_cvar = []

    for i in range(len(vals) - 1):
        #look at values between
        sigma_range = np.linspace(vals[i], vals[i+1],200)
        for sigma in sigma_range:
            cvar_val = sigma - 1/(1-alpha) * np.sum([probs[i] * max(0,sigma - vals[i]) for i in range(len(vals)) ])
            all_cvar.append(cvar_val)
            all_sigma.append(sigma)
            if cvar_val > max_cvar:
                max_cvar = cvar_val
                sigma_star = sigma
    return max_cvar, sigma_star, all_cvar, all_sigma


probs = [1/3, 1/3, 1/3]
vals = [ 0.33333333, -0.16666667, -0.33333333]
alpha = 0.5
cvar, var, all_cvar, all_sigma = brute_force_solve_cvar(vals, probs, alpha)
print("cvar", cvar, "var", var)
plt.plot(all_sigma, all_cvar)
plt.show()
