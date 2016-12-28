# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 10:49:42 2016

@author: minori
"""

import pandas as pd
x = pd.Series([2,3,4,1])
y = pd.Series([5,6,7,8], index = ['a','b','c','d'])
print(y['a'])
print(y[['a','b']])
age = {'Tim':29, 'Jim':32, 'Pam':67, 'Qoo':20}
age_series = pd.Series(age)
print(age_series)

data = {'name':['Tim', 'Jim', 'Pam', 'Sam'],
        'age' :[29, 12, 43, 34],
        'ZIP' :['09249', '02340', '63465', '76553']
                }
data_fram = pd.DataFrame(data, columns = ['name', 'age', 'ZIP'])

print(data_fram)                
print(data_fram['name'])
print(data_fram.name)

print(y.index)
print(sorted(y.index))
print(y.reindex(sorted(y.index)))

w = pd.Series([1,2,3,4], index = ['a','b','c','d'])
z = pd.Series([5,6,7,8], index = ['c','d','e','f'])
print(w+z)


table = pd.DataFrame(columns = ('name', 'age'))
table.loc[1] = 'James', 22
table.loc[2] = 'Jess', 32
print(table)
print(table.columns)