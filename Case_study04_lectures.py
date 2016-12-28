# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 10:49:42 2016

@author: minori
"""

import pandas as pd
import numpy as np
whisky = pd.read_csv('whiskies.txt')
whisky['Region'] = pd.read_csv('regions.txt')
print(whisky.head())
print(whisky.iloc[0:10])
#print(whisky.loc[0:9] == whisky.iloc[0:10])
print(whisky.iloc[5:10, 0:5])
print(whisky.columns)
flavors = whisky.iloc[:, 2:14]
print(flavors)
corr_flavors = pd.DataFrame.corr(flavors)
print(corr_flavors)

import pylab as plt

#plt.figure(figsize = (10,10))
#plt.pcolor(corr_flavors)
#plt.colorbar()
#plt.savefig('corr_flavors.pdf')
#
corr_whisky = pd.DataFrame.corr(flavors.transpose())
#plt.figure(figsize = (10,10))
#plt.pcolor(corr_whisky)
#plt.axis('tight')
#plt.colorbar()
#plt.savefig('corr_whisky.pdf')


from sklearn.cluster.bicluster import SpectralCoclustering
model = SpectralCoclustering(n_clusters = 6, random_state = 0)
model.fit(corr_whisky)
print(model.rows_)
print(np.sum(model.rows_, axis = 1))
print(np.sum(model.rows_, axis = 0))
print(model.row_labels_)


whisky['Group'] = pd.Series(model.row_labels_, index = whisky.index)
whisky = whisky.ix[np.argsort(model.row_labels_)]
whisky  = whisky.reset_index(drop = True)
correlation = pd.DataFrame.corr(whisky.iloc[:,2:14].transpose())
correlation = np.array(correlation)


plt.figure(figsize = (14,7))
plt.subplot(121)
plt.pcolor(corr_whisky)
plt.colorbar()
plt.title('Original')
plt.axis('tight')
plt.subplot(122)
plt.pcolor(correlation)
plt.colorbar()
plt.title('Rerranged')
plt.axis('tight')
plt.savefig('correlation.pdf')

'''
data = pd.Series([1,2,3,4])
data = data.ix[[3,0,1,2]]
#data = data.reset_index(drop=True) 
'''





