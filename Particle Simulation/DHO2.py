# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 09:16:37 2022
Lutz Lehmann. Cannot get rk4 to solve for position of orbiting body in python.
https://stackoverflow.com/questions/53645649/cannot-get-rk4-to-solve-for-position-of-orbiting-body-inpython,
2018. Online - Accessed 5 Dec 2021. Provided RK4Integrate Function/Array splicing method
that made the code work. Used there G and M parameters.
@author: Ezra
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import maxwell

N=1000
b=2.972*10**-21
k1=1.530*10**-17
k=1.3806*10**-23
T=305.4
M=1.443160768573*10**-25
a=np.sqrt((k*T)/M)
t = np.linspace(0,.01,100)

def ran(x,y):
    unitArray = np.random.rand(x,y)
    theta = 2*np.pi*unitArray[0,:]
    u = unitArray[1,:]*2-1
    return np.array([theta, u])


def unitVector(theta, u):
    x = np.sqrt(1-u**2)*np.cos(theta)
    y = np.sqrt(1-u**2)*np.sin(theta)
    z = u
    return np.array([x,y,z])

unitArray = ran(2,N)
unitArray1 = ran(2,N)

s=unitVector(unitArray[0,:],unitArray[1,:])
s1=unitVector(unitArray1[0,:],unitArray1[1,:])

mu, sigma = 0, 0.01 # mean and standard deviation
randGaussian = np.random.normal(mu, sigma, N)
randMaxwell = maxwell.rvs(loc=0,scale = a, size=N)

inPosition = randGaussian*s
inVelocity = randMaxwell*s1

# vector1 = np.column_stack((inPosition.T,inVelocity.T))

vector= np.concatenate((inPosition,inVelocity),0)

def func(intialArray):
    x,y,z,vx,vy,vz = intialArray
    r=np.array([x,y,z])
    v=np.array([vx,vy,vz])
    rdot = v
    vdot = -(k1/M)*r-(b/M)*v
    rdot=rdot.T
    vdot=vdot.T
    vector = np.concatenate((rdot,vdot),0)
    return vector

def RK4(f,r,h):
    k1 = h*f(r)
    k2 = h*f(r+.5*k1)
    k3 = h*f(r+.5*k2)
    k4 = h*f(r+k3)
    return r + (k1+2*k2+2*k3+k4)/6
    
def RK4integrate(f, y0, tspan):
    u = np.zeros([len(tspan),len(y0)])
    u[0,:]=y0
    for k in range(1, len(tspan)):
        u[k,:] = RK4(f, u[k-1], tspan[k]-tspan[k-1])
    return u

solTens = np.zeros([len(t),len(vector),len(vector.T)])
for i in range(len(vector.T)):
    solTens[:,:,i] = RK4integrate(func, vector[:,i], t)
  
    
def Energy(solTens):
    energyTens=np.zeros([len(t),len(vector),len(vector.T)])
    for i in range(len(solTens[1,1,:])):
        for k in range(len(solTens[:,1,:])):
            V=.5*k1*(solTens[k,0,i]**2+solTens[k,1,i]**2+solTens[k,2,i]**2)
            T=.5*M*(solTens[k,3,i]**2+solTens[k,4,i]**2+solTens[k,5,i]**2)
            energyTens[k,:,i]=T+V
    return energyTens


energyTens = Energy(solTens)
energyArray = energyTens[:,1,:]
energyArray = np.sum(energyArray, axis = 1)

T=(energyArray*2)/(3*k)

table = np.column_stack((t,T))

df = pd.DataFrame(table)
df.to_excel(excel_writer='GroupB_PH486_DHO2.xlsx')

plt.figure(0)
plt.grid(True)
plt.yscale('log')
plt.xlabel('time [s]')
plt.ylabel('temperature [K]')
plt.plot(t,T)

plt.figure(1)
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('temperature [K]')
plt.plot(t,T)

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


