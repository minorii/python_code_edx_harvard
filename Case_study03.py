# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 00:47:01 2016

@author: minori
"""

import pandas as pd
import numpy as np
import random
data = pd.read_csv('https://s3.amazonaws.com/demo-datasets/wine.csv')
numeric_data = data.drop("color", axis=1)
#numeric_data = data.iloc[:, data.columns != 'color']
#numeric_data = data.ix[:, data.columns != 'color']
numeric_data = (numeric_data - np.mean(numeric_data, axis=0)) / np.std(numeric_data, axis=0)
#numeric_data.mean()

import sklearn.decomposition
pca = sklearn.decomposition.PCA(2)
principal_components = pca.fit(numeric_data).transform(numeric_data)

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
observation_colormap = ListedColormap(['red', 'blue'])
x = principal_components[:,0]# Enter your code here!
y = principal_components[:,1]# Enter your code here!

plt.title("Principal Components of Wine")
plt.scatter(x, y, alpha = 0.2,
    c = data['high_quality'], cmap = observation_colormap, edgecolors = 'none')
plt.xlim(-8, 8); plt.ylim(-8, 8)
plt.xlabel("Principal Component 1"); plt.ylabel("Principal Component 2")
plt.show()

def accuracy(predictions, outcomes):
    """
    Finds the percent of predictions that equal outcomes.
    """
    return 100*np.mean(predictions == outcomes)
#    return sum(predictions == outcomes)/len(predictions)

#print(accuracy(np.zeros(len(data.ix[:,'high_quality'])), data.ix[:,'high_quality']))
print(accuracy(0, data["high_quality"]))

#print(100*np.mean(sk_predictions == my_predictions))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(numeric_data, data['high_quality'])
# Enter your code here!
library_predictions = knn.predict(numeric_data)
print(accuracy(library_predictions, data["high_quality"]))

n_rows = data.shape[0]
# Enter your code here.
#random.seed(12)
selection = random.sample(range(n_rows),10)

import week3
predictors = np.array(numeric_data)
outcomes = np.array(data["high_quality"])
my_predictions = np.array([week3.knn_predict(p, predictors, outcomes, k = 5) 
                        for p in predictors[selection]])# Enter your code here!
percentage = accuracy(my_predictions, outcomes[selection])# Enter your code here!
print(percentage)



