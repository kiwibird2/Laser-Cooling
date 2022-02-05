# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 11:28:22 2022

@author: Ezra

How to generate negative random value in python. BluePeppers. StackExchange. https://stackoverflow.com/questions/10579518/how-to-generate-negative-random-value-in-python/10579562. Accesed 29 Jan 2022

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell
from mpl_toolkits.mplot3d import axes3d

k=1.3806*10**-23
T=300
M=1.4431578*10**-25
a=np.sqrt((k*T)/M)

"""
generate random theta and u values
"""

def ran(x,y):
    unitArray = np.random.rand(x,y)
    theta = 2*np.pi*unitArray[0,:]
    u = unitArray[1,:]*2-1
    return np.array([theta, u])

"""
generate unit vector componenets
"""

def unitVector(theta, u):
    x = np.sqrt(1-u**2)*np.cos(theta)
    y = np.sqrt(1-u**2)*np.sin(theta)
    z = u
    return np.array([x,y,z])

"""
create two Arrays for the intial position and velocity
"""

unitArray = ran(2,10)
unitArray1 = ran(2,10)

s=unitVector(unitArray[0,:],unitArray[1,:])
s1=unitVector(unitArray1[0,:],unitArray1[1,:])

"""
make random gaussian and maxwell distributions
"""

mu, sigma = 0, 0.05 # mean and standard deviation
randGaussian = np.random.normal(mu, sigma, 10)
randMaxwell = maxwell.rvs(loc=0,scale = a, size=10)

"""
create a random intial position and intial velcoity
"""

inPosition = randGaussian*s
inVelocity = randMaxwell*s1

"""
transpose arrays for proper formating
"""

inPositionTran=np.transpose(inPosition)
inVelocityTran=np.transpose(inVelocity)

"""
put the array together
"""

vector1 = np.column_stack((inPosition,inVelocity))

"""
Uncomment to see
Visualsization of random unit vectors
Viualization of random position values
"""

u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
b = np.cos(u)*np.sin(v)
n = np.sin(u)*np.sin(v)
m = np.cos(v)

fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
ax.scatter(s[0,:], s[1,:], s[2,:], s=100, c='b', zorder=10)
ax.plot_wireframe(b,n,m, color="k")


fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
ax.scatter(inPosition[0,:], inPosition[1,:], inPosition[2,:], s=100, c='b', zorder=10)


