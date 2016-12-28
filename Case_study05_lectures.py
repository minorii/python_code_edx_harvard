# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 08:47:55 2016

@author: minori
"""

import pandas as pd
import pylab as plt
import numpy as np

birddata = pd.read_csv('bird_tracking.csv')
birddata.info()
print(birddata.head())

bird_names = pd.unique(birddata.bird_name)
print(bird_names)

'''
plt.figure(figsize = (7,7))
for bird_name in bird_names:
    ix = birddata.bird_name == bird_name
    x, y = birddata.longitude[ix], birddata.latitude[ix]    
    plt.plot(x, y, '.', label = bird_name)
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.legend(loc = 'lower right')
plt.savefig('3traj.pdf')
#'''

ix = birddata.bird_name == 'Eric'
speed = birddata.speed_2d[ix]

print(np.isnan(speed).any())
print(np.sum(np.isnan(speed)))
ind = np.isnan(speed)
#print(ind)
#print(~ind)

'''
plt.hist(speed[~ind])
plt.savefig('bird_Eric_speed.pdf')

plt.figure(figsize = (8,4))
speed = birddata.speed_2d[birddata.bird_name == 'Eric']
ind = np.isnan(speed)
plt.hist(speed[~ind], bins = np.linspace(0, 30, 20), normed = True)
plt.xlabel('20 speed (m/x)')
plt.ylabel('Frequency')
#'''

#we do not need to deal with NAN explictly
'''
birddata.speed_2d.plot(kind = 'hist', range = [0, 30])
plt.xlabel('2D speed')
plt.savefig('pd_hist.pdf')
#'''

print(birddata.columns)
print(birddata.date_time[0:3])

import datetime

#print(datetime.datetime.today())
#time_1 = datetime.datetime.today()
#time_2 = datetime.datetime.today()
#print(type(time_1-time_2))
#date_str = birddata.date_time[0]
#print(date_str)
#print(date_str[:-3])
#print(datetime.datetime.strptime(date_str[:-3], '%Y-%m-%d %H:%M:%S'))
#print(type(datetime.datetime.strptime(date_str[:-3], '%Y-%m-%d %H:%M:%S')))

timestamps = []
for k in range(len(birddata)):
    timestamps.append(datetime.datetime.strptime(
                    birddata.date_time.iloc[k][:-3], '%Y-%m-%d %H:%M:%S'))
print(timestamps[0:3])
birddata['timestamp'] = pd.Series(timestamps, index = birddata.index)
#print(birddata.head())
times = birddata.timestamp[birddata.bird_name == 'Eric']
elapsed_time = [time - times[0] for time in times]
print(elapsed_time[0:3])
#print(elapsed_time[1000]/datetime.timedelta(days = 1))
#print(elapsed_time[1000]/datetime.timedelta(hours = 1))
plt.plot(np.array(elapsed_time) / datetime.timedelta(days = 1))
plt.xlabel('Observation')
plt.ylabel('Elapsed time(days)')
plt.savefig('timeplot.pdf')


data = birddata[birddata.bird_name == 'Eric']
time = data.timestamp
elapsed_time = [time - times[0] for time in times]
elapsed_days = np.array(elapsed_time) / datetime.timedelta(days = 1)

next_day = 1
inds = []
daily_mean_speed = []
for (i, t) in enumerate(elapsed_days):
    if t < next_day:
        inds.append(i)
    else:
        daily_mean_speed.append(np.mean(data.speed_2d[inds]))
        next_day += 1
        inds = []

plt.figure(figsize = (8, 6))
plt.plot(daily_mean_speed)
plt.xlabel('Day')
plt.ylabel('Mean speed (m/s)')
plt.savefig('dms.pdf')
        

import cartopy.crs as ccrs
import cartopy.feature as cfeature

proj = ccrs.Mercator()

plt.figure(figsize = (10, 10))
ax = plt.axes(projection = proj)
ax.set_extent((-25.0, 20.0, 52.0, 10.0))
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle = ':')

for name in bird_names:
    ix = birddata['bird_name'] == name
    x, y = birddata.longitude[ix], birddata.latitude[ix]
    ax.plot(x, y, '.', transform = ccrs.Geodetic(), label = name)
    
plt.legend(loc = 'upper left')
plt.savefig('map.pdf')



# First, use `groupby` to group up the data.
grouped_birds = birddata.groupby("bird_name")

# Now operations are performed on each group.
mean_speeds = grouped_birds.speed_2d.mean()

# The `head` method prints the first 5 lines of each bird.
print(grouped_birds.head())

# Find the mean `altitude` for each bird.
# Assign this to `mean_altitudes`.
mean_altitudes = birddata.groupby("bird_name").altitude.mean()## YOUR CODE HERE ##
                    

# Convert birddata.date_time to the `pd.datetime` format.
birddata.date_time = pd.to_datetime(birddata.date_time)

# Create a new column of day of observation
birddata["date"] = birddata.date_time.dt.date

# Check the head of the column.
print(birddata.date.head())

grouped_bydates = birddata.groupby('date')## YOUR CODE HERE ##
mean_altitudes_perday = grouped_bydates.altitude.mean()## YOUR CODE HERE ##


grouped_birdday = birddata.groupby(['bird_name', 'date'])## YOUR CODE HERE ##
mean_altitudes_perday = grouped_birdday.altitude.mean()

# look at the head of `mean_altitudes_perday`.
print(mean_altitudes_perday.head())


eric_daily_speed  = birddata.groupby(['bird_name', 'date']).speed_2d.mean()['Eric']# Enter your code here.
sanne_daily_speed = birddata.groupby(['bird_name', 'date']).speed_2d.mean()['Sanne']# Enter your code here.
nico_daily_speed  = birddata.groupby(['bird_name', 'date']).speed_2d.mean()['Nico']# Enter your code here.
                     
plt.figure(figsize = (10, 10))
eric_daily_speed.plot(label="Eric")
sanne_daily_speed.plot(label="Sanne")
nico_daily_speed.plot(label="Nico")
plt.legend(loc="upper left")
plt.show()


























