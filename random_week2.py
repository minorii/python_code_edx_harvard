# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 12:43:52 2016

@author: minori
"""

import random
import pylab as plt
import numpy as np

rolls = []
for k in range(1000):
    rolls.append(random.choice([1,2,3,4,5,6]))
plt.hist(rolls, bins = np.linspace(0.5, 6.5, 7))

#%%
ys = []

for rep in range(1000):
    y = 0
    for i in range(10):
        y += random.choice([1,2,3,4,5,6])
    ys.append(y)
    
plt.hist(ys, bins = np.linspace(2, 63, 61))

#%%
print(np.random.random((2,3)))
np.random.normal(0,1, (4,3))
print(np.random.uniform(low = 0.0, high = 2, size = (5, 5, 2)))

#%%
import time
start_time = time.clock()
x = np.random.randint(1, 7, (1000,10))
x.shape
Y = np.sum(x, axis = 1)
plt.hist(Y, bins = 20)
end_time = time.clock()
print(start_time - end_time)

#%%
delta_X = np.random.normal(0,1,(2,1000))
X = np.cumsum(delta_X, axis = 1)
X_0 = np.array([[0], [0]])
X = np.concatenate((X_0, X), axis = 1)
plt.plot(X[0], X[1], 'g-', markersize = 0.1)
plt.savefig('random2.pdf')