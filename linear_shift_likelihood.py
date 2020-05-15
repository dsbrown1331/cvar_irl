import numpy as np

#interesting...
#if we want a likelihood that is shift invariant then using the exponential is good!
#If we subtract a certain amount from each feature this ammounts to accumulating these differences and just subtracting something from the feature counts
#this is the same as just subtracting a constant from the reward at each step which should keep optimal policies optimal, assuming
#we have an infinite loop at the terminal and subtract/add the feature to the terminal all zero feature too.
#changing Beta is a positive scaling and that has a big impact on likelihood.

fcounts = np.array([[2,0,3],
                    [2,3,0]])

w = np.array([0,0,1])
print("w", w)
#evaluate likelihood
Qs = np.dot(fcounts, w)
print("Qs", Qs)
beta = 1.0

log_likelihood = beta * Qs[0] - np.log(np.sum(np.exp(beta * Qs)))
print("original log likleihood", log_likelihood)


#now change the fcounts
fcounts -= 2

#evaluate likelihood
Qs = np.dot(fcounts, w)
print("Qs", Qs)
beta = 1.0

log_likelihood = beta * Qs[0] - np.log(np.sum(np.exp(beta * Qs)))
print("likelihood random", log_likelihood)


#now add a constant to each of the weights
w = np.array([0,0,1]) + 3
print("w", w)
#evaluate likelihood
Qs = np.dot(fcounts, w)
print("Qs", Qs)
beta = 1.0

log_likelihood = beta * Qs[0] - np.log(np.sum(np.exp(beta * Qs)))
print("equal shift likelhood", log_likelihood)