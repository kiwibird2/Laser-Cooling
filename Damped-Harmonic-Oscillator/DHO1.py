# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 22:16:14 2022

CDT Kim, Beomjoon. Company B3. Assistance given through verbal instruction. Told me that I was missing the absolute value on our second velocity in both function. West Point, NY. 20 Jan 2021.

Coffey, Tonya. Enery of Damped Oscillators and Quality Factors. Youtube. https://www.youtube.com/watch?v=-VHj6d_RgDg. 19 Oct 2020.
Helped me fix kinetic energy equation. Used k instead of mass.

@author: Ezra
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

"""
Code Preamble: Variable definitions:
    b = linear damping term
    c = quadratic damping term
    k1 = spring constant
    k2 = spring constant
    m = mass of cart
"""

b=0.356
c=0.432
k1=1.2
k2=1.3
m=0.75

"""
Import position and velcoity data:
"""

position = pd.read_csv(r'C:\Users\Ezra\Desktop\PH486\DHO1\position.csv')
velocity = pd.read_csv(r'C:\Users\Ezra\Desktop\PH486\DHO1\velocity.csv')

t = position['t(s)']
xActual = position['x(m)']
vActual = velocity['v(m/s)']

"""
Function for ideal case:
    returns matrix of position and velocity
"""

def func(u,t,b,m,c,k1,k2):
    x=u[0]
    z=u[1]
    xdot = z
    zdot = -((k1+k2)/m)*x-(c/m)*z*np.abs(z)-(b/m)*z
    return np.array([xdot,zdot],float)

"""
Function for experimetal fit:
    returns matrix of position and velocity
"""

def fit(u,t,m,k1,k2):
    x=u[0]
    z=u[1]
    xdot = z
    zdot = -((k1+k2)/m)*x-(.235/m)*z*np.abs(z)-(.295/m)*z
    return np.array([xdot,zdot],float)

"""
Inital conditions for three trial cases:
    u0:max displacement, no velocity
    u1:some displacement, some velocity
    u2:no displacement, max velocity
"""

u0 = np.array([1,0])
u1 = np.array([.5,.5])
u2 = np.array([0,1])

"""
Called odeint to solve ideal case and generate fit function
"""

Usol = odeint(func,u0,t,args=(b,m,c,k1,k2))
Usol1 = odeint(func,u1,t,args=(b,m,c,k1,k2))
Usol2 = odeint(func,u2,t,args=(b,m,c,k1,k2))
Ufit = odeint(fit,u0,t,args=(m,k1,k2))

"""
Calculate potential, kinetic, and total energy
"""

Potential = .5*k1*(Usol[:,0])**2+.5*k2*(Usol[:,0])**2
Kinetic = .5*m*(Usol[:,1])**2
TotEnergy = Potential+Kinetic

Potential1 = .5*k1*(Usol1[:,0])**2+.5*k2*(Usol1[:,0])**2
Kinetic1 = .5*m*(Usol1[:,1])**2
TotEnergy1 = Potential1+Kinetic1

Potential2 = .5*k1*(Usol2[:,0])**2+.5*k2*(Usol2[:,0])**2
Kinetic2 = .5*m*(Usol2[:,1])**2
TotEnergy2 = Potential2+Kinetic2



"""
Sum of least squares
"""
r = (xActual-Ufit[:,0])**2
r2 = sum(r)
print(r2)


plt.figure(1)
plt.plot(t,Usol[:,0],'r')
plt.plot(t,Usol[:,1],'b--')
# plt.title('damped harmonic oscillator\nIC: x=1m  dx/dt=0m/s' )
plt.xlabel('time [s]')
plt.ylabel('position[m]\n speed[m/s]')
plt.grid('true')
plt.legend(['x [m]','dx\dt [m/s]'])
# plt.figtext(.70,.40, 'IC: [1m,0m/s]')
# plt.figtext(.70,.35, 'b = 0.356 kg/s' )
# plt.figtext(.70,.30, 'c = 0.432 kg/s')
# plt.figtext(.70,.25, 'k1 = 1.20 N/m')
# plt.figtext(.70,.20, 'k2 = 1.30 N/m')
# plt.figtext(.70,.15, 'm = 0.75 kg')
plt.show()


plt.figure(2)
plt.plot(t,Usol1[:,0],'r')
plt.plot(t,Usol1[:,1],'b--')
# plt.title('damped harmonic oscillator\nIC: x=1m  dx/dt=0m/s' )
plt.xlabel('time [s]')
plt.ylabel('position[m]\n speed[m/s]')
plt.grid('true')
plt.legend(['x [m]','dx\dt [m/s]'])
# plt.figtext(.70,.40, 'IC: [.5m,.5m/s]')
# plt.figtext(.70,.35, 'b = 0.356 kg/s' )
# plt.figtext(.70,.30, 'c = 0.432 kg/s')
# plt.figtext(.70,.25, 'k1 = 1.20 N/m')
# plt.figtext(.70,.20, 'k2 = 1.30 N/m')
# plt.figtext(.70,.15, 'm = 0.75 kg')
plt.show()


plt.figure(3)
plt.plot(t,Usol2[:,0],'r')
plt.plot(t,Usol2[:,1],'b--')
# plt.title('damped harmonic oscillator\nIC: x=1m  dx/dt=0m/s' )
plt.xlabel('time [s]')
plt.ylabel('position[m]\n speed[m/s]')
plt.grid('true')
plt.legend(['x [m]','dx\dt [m/s]'])
# plt.figtext(.70,.40, 'IC: [1m,0m/s]')
# plt.figtext(.70,.35, 'b = 0.356 kg/s' )
# plt.figtext(.70,.30, 'c = 0.432 kg/s')
# plt.figtext(.70,.25, 'k1 = 1.20 N/m')
# plt.figtext(.70,.20, 'k2 = 1.30 N/m')
# plt.figtext(.70,.15, 'm = 0.75 kg')
plt.show()


plt.figure(4)
plt.plot(t,xActual,'k.')
plt.plot(t,Ufit[:,0],'r')
plt.grid('true')
plt.xlabel('time [s]')
plt.ylabel('position[m]')
plt.grid('true')
plt.legend(['data','fit'])
# plt.figtext(.70,.70, 'b = 0.295 kg/s' )
# plt.figtext(.70,.65, 'c = 0.235 kg/s')
# plt.figtext(.70,.60, 'k1 = 1.20 N/m')
# plt.figtext(.70,.55, 'k2 = 1.30 N/m')
# plt.figtext(.70,.50, 'm = 0.75 kg')
plt.show()

plt.figure(5)
plt.plot(t,Potential,'b')
plt.plot(t,Kinetic,'r')
plt.plot(t,TotEnergy,'purple')
plt.grid('true')
plt.xlabel('time [s]')
plt.ylabel('Energy [J]')
plt.grid('true')
plt.legend(['potential','kinetic','total'])
# plt.figtext(.70,.70, 'b = 0.295 kg/s' )
# plt.figtext(.70,.65, 'c = 0.235 kg/s')
# plt.figtext(.70,.60, 'k1 = 1.20 N/m')
# plt.figtext(.70,.55, 'k2 = 1.30 N/m')
# plt.figtext(.70,.50, 'm = 0.75 kg')
plt.show()

plt.figure(5)
plt.plot(t,Potential1,'b')
plt.plot(t,Kinetic1,'r')
plt.plot(t,TotEnergy1,'purple')
plt.grid('true')
plt.xlabel('time [s]')
plt.ylabel('Energy [J]')
plt.grid('true')
plt.legend(['potential','kinetic','total'])
# plt.figtext(.70,.70, 'b = 0.295 kg/s' )
# plt.figtext(.70,.65, 'c = 0.235 kg/s')
# plt.figtext(.70,.60, 'k1 = 1.20 N/m')
# plt.figtext(.70,.55, 'k2 = 1.30 N/m')
# plt.figtext(.70,.50, 'm = 0.75 kg')
plt.show()


plt.figure(5)
plt.plot(t,Potential2,'b')
plt.plot(t,Kinetic2,'r')
plt.plot(t,TotEnergy2,'purple')
plt.grid('true')
plt.xlabel('time [s]')
plt.ylabel('Energy [J]')
plt.grid('true')
plt.legend(['potential','kinetic','total'])
# plt.figtext(.70,.70, 'b = 0.295 kg/s' )
# plt.figtext(.70,.65, 'c = 0.235 kg/s')
# plt.figtext(.70,.60, 'k1 = 1.20 N/m')
# plt.figtext(.70,.55, 'k2 = 1.30 N/m')
# plt.figtext(.70,.50, 'm = 0.75 kg')
plt.show()