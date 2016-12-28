# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 01:40:02 2016

@author: minori
"""

import numpy as np
import pylab as plt
#x = np.linspace(0,10,20)
x = np.logspace(-1,1,20)
y1 = x**2.0
y2 = x**1.5
#plt.plot(x, y1, 'gd-', linewidth = 2, markersize = 19,
#         label = 'First')
#plt.plot(x, y2, 'bo-')
plt.loglog(x, y1, 'gd-', linewidth = 2, markersize = 19,
         label = 'First')
plt.loglog(x, y2, 'bo-')
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.axis([-0.5,10.5,-5,105])
plt.legend(loc = 'upper left')
plt.savefig('plot.pdf')
#semilogx()
#semilogy()
#loglog()

#%%
import numpy as np
import pylab as plt
np.random.seed(0)
x = np.random.normal(size = 1000)
plt.hist(x, normed = True, bins = np.linspace(-4,4,21))
#%%
import numpy as np
import pylab as plt
np.random.seed(0)
x = np.random.gamma(2, 3, 10000)
plt.figure()
plt.subplot(221)
plt.hist(x, bins = 30)
plt.subplot(222)
plt.hist(x, bins = 30, normed = True)
plt.subplot(223)
plt.hist(x, cumulative = True)
plt.subplot(224)
plt.hist(x, bins = 30, normed = True, cumulative = True
         , histtype = 'step')
plt.savefig('hist.pdf')