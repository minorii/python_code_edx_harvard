# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 00:18:49 2016

@author: minori
"""
from collections import Counter
import numpy as np
import pylab as plt
'''
x = [1,2,3,1,2,3,3,3,3,3]
print(Counter(x))
count_x = Counter(x)
max_index = max(count_x)
max_count = max(count_x.values())
'''
def majority_votes(votes):
    vote_counts = Counter(votes)
    winner = []
    for vote, count in vote_counts.items():
        #    print(vote, item)
        if count == max(vote_counts.values()):
            winner.append(vote)
    return np.random.choice(winner)



import scipy.stats as ss

def majority_votes_fast(votes):
    mode, count = ss.mstats.mode(votes)
    return mode

def distance(a, b):
    return np.sqrt(sum((a[i]-b[i])**2 for i in range(len(a))))
'''
print(majority_votes(x))
print(majority_votes_fast(x))

points = np.array([[1,1], [1,2], [1,3], [2,1], [2,2], [2,3], [3,4], [6,4], [5,5]])
p = np.array([2.5, 2])
outcomes = np.array([0,0,0,0,1,1,1,1,1])
'''

def find_nearsest_neighbors(p, points, k = 5):
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p, points[i])
    ind = np.argsort(distances)
    return ind[0:k]
    
def knn_predict(p, points, outcomes, k = 5):
    ind = find_nearsest_neighbors(p, points, k)
    return majority_votes(outcomes[ind])
    
def generate_synth_data(n = 50):
    points = np.concatenate((ss.norm(0,1).rvs((n,2)), 
                             ss.norm(1,1).rvs((n,2))), axis = 0)
    outcomes = np.concatenate((np.repeat(0, n), 
                               np.repeat(1, n)), axis = 0)
    return (points, outcomes)
    
'''
ind = find_nearsest_neighbors(p, points, k = 2)
print(points[ind])
print(knn_predict(p, points, outcomes, k = 2))

n = 20
points, outcomes = generate_synth_data(n)


plt.figure()
plt.plot(points[:n,0], points[:n, 1], 'ro')
plt.plot(points[n:,0], points[n:, 1], 'bo')
plt.savefig('bivardata.pdf')



plt.figure()
plt.plot(points[:,0], points[:,1], 'ro')
plt.plot(p[0], p[1], 'bo')
plt.axis([0.5, 6.5, 0.5, 5.5])
'''

def make_prediction_grid(predictors, outcomes, limits, h, k):
    (x_min, x_max, y_min, y_max) = limits
    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xs, ys)
    
    prediction_grid = np.zeros(xx.shape, dtype = int)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            p = np.array([x,y])
            prediction_grid[j,i] = knn_predict(p, predictors, outcomes, k)
    return (xx, yy, prediction_grid)
    
#(predictors, outcomes) = generate_synth_data()
#print(predictors.shape)
#print(outcomes.shape)
def plot_prediction_grid (xx, yy, prediction_grid, filename):
    """ Plot KNN predictions for every point on the grid."""
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green"])
    plt.figure(figsize =(15,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, 
                   alpha = 0.5, rasterized = True)
    plt.scatter(predictors[:,0], predictors [:,1], c = outcomes, 
                cmap = observation_colormap, s = 100)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    plt.savefig(filename)

    '''
#k = 50
#filename = 'knn_synth_5.pdf'
#limits = (-3,4, -3,4)
#h = 0.1
#for k in range(5, 50, 5):
#    (xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits, h, k)
#    plot_prediction_grid(xx, yy, prediction_grid, filename)
#    plt.savefig('knn_prediction'+str(k)+' .pdf')
'''

from sklearn import datasets
iris = datasets.load_iris()

predictors = iris.data[:, 0:2]
outcomes = iris.target

'''
plt.plot(predictors[outcomes == 0][:,0], predictors[outcomes == 0][:,1], 'ro')
plt.plot(predictors[outcomes == 1][:,0], predictors[outcomes == 1][:,1], 'yo')
plt.plot(predictors[outcomes == 2][:,0], predictors[outcomes == 2][:,1], 'bo')
plt.savefig('iris.pdf')

k = 5
filename = 'iris_grid_50.pdf'
limits = (4,8, 1.5,4.5)
h = 0.1
(xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx, yy, prediction_grid, filename)
#plt.savefig('iris_grid_'+str(k)+' .pdf')
#'''

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(predictors, outcomes)
sk_predictions = knn.predict(predictors)

my_predictions = np.array([knn_predict(p, predictors, outcomes, 5) 
                            for p in predictors])



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




