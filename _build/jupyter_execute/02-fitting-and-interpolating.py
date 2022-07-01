#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# # Our familiar example from previous weeks
# 
# Compound interest calculator with annual contributions
# 
# * p = principal
# * r = annual interest rate in percent
# * y = year of the balance
# * c = annual contribution (made at the start of the year)
# 
# $$\text{Balance}(y) = p(1 + r)^y + c\left[\frac{(1 + r)^{y+1} - (1 + r)}{r} \right]$$

# In[ ]:


def investment_balance(principal,rate,year,contribution):
    p = principal
    r = rate/100
    y = year
    c = contribution
    balance = p*(1 + r)**y + c*( ((1 + r)**(y+1) - (1 + r)) / r )
    return balance


# In[ ]:


ratesnp = np.linspace(1,25,25)


# In[ ]:


ratesnp


# In[ ]:


ib = investment_balance(1000,ratesnp,10,1000)


# In[ ]:


plt.plot(ratesnp,ib,'ro')


# In[ ]:


np.polyfit(ratesnp, ib, 3)


# In[ ]:


p = np.poly1d(np.polyfit(ratesnp, ib, 3))


# In[ ]:


p


# In[ ]:


t = np.linspace(0, 25, 200)


# In[ ]:


plt.plot(ratesnp,ib,'ro',t,p(t))


# In[ ]:


p.coef


# In[ ]:


def f(x):
    return 9.68449317e-01 * (x**3) + 1.11578231e+01 * (x**2) + 7.12885916e+02 * x + 1.09010025e+04


# In[ ]:


plt.plot(t,f(t))


# In[ ]:


import ipywidgets


# In[ ]:


def f(order=1):
    t = np.linspace(0, 25, 200)
    p = np.poly1d(np.polyfit(ratesnp, ib, order))
    plt.plot(ratesnp,ib,'ro',t,p(t))

ipywidgets.interact(f,order=(1,10))


# In[ ]:


x = np.linspace(0,5*np.pi,50)
noise = (np.random.random(50)*2 - 1) * 0.1
y = np.sin(x) + noise
plt.plot(x,y,'ro')


# In[ ]:


def f(order=1):
    t = np.linspace(0, 5*np.pi, 200)
    p = np.poly1d(np.polyfit(x,y, order))
    plt.plot(x,y,'ro',t,p(t))

ipywidgets.interact(f,order=(1,10))


# In[ ]:


x = np.linspace(-3*np.pi,3*np.pi,20)
noise = (np.random.random(20)*2 - 10) * 200
y = x**3-6*x**2 + noise
plt.plot(x,y,'ro')


# In[ ]:


def f(order=1):
    p = np.poly1d(np.polyfit(x,y, order))
    t = np.linspace(-3*np.pi, 3*np.pi, 500)
    plt.plot(x,y,'ro',t,p(t))

ipywidgets.interact(f,order=(1,20))


# # On to SciPy

# In[ ]:


from scipy.optimize import curve_fit


# In[ ]:


# use the same data from above
x = np.linspace(-3*np.pi,3*np.pi,20)
noise = (np.random.random(20)*2 - 10) * 200
y = x**3-6*x**2 + noise

# must provide a model function, with independent variable as the first parameter
def m(x,a,b):
    return a + b*x

# call scipy's curve fitting routine
curve_fit(m,x,y)

# returns the parameters and the covariance for the estimated parameters


# In[ ]:


# use the same data from above
x = np.linspace(-3*np.pi,3*np.pi,20)
noise = (np.random.random(20)*2 - 10) * 200
y = x**3-6*x**2 + noise

# must provide a model function, with independent variable as the first parameter
def m(x,a,b):
    return a + b*x

# call scipy's curve fitting routine
c = curve_fit(m,x,y)

plt.plot(x,y,'ro',x,m(x,c[0][0],c[0][1]))


# In[ ]:


##
## EXERCISE -> change this so that the curve fit is for a 3rd order polynomial
##

# use the same data from above
x = np.linspace(-3*np.pi,3*np.pi,20)
noise = (np.random.random(20)*2 - 10) * 200
y = x**3-6*x**2 + noise

# must provide a model function, with independent variable as the first parameter
def m(x,a,b):
    return a + b*x

# call scipy's curve fitting routine
c = curve_fit(m,x,y)

plt.plot(x,y,'ro',x,m(x,c[0][0],c[0][1]))


# In[ ]:


# %load xercise-3curve.py


# # Interpolation

# In[ ]:


from scipy import interpolate


# In[ ]:


x = np.linspace(0,5*np.pi,20)
noise = (np.random.random(20)*2 - 1) * 0.1
y = np.sin(x) + noise
plt.plot(x,y,'ro')


# In[ ]:


linear_interp = interpolate.interp1d(x, y)

xlin = np.linspace(0, 5*np.pi, 200)
ylin = linear_interp(xlin)

plt.plot(x,y,'ro')
plt.plot(xlin,ylin,'b-')


# In[ ]:


localinterp = interpolate.interp1d(x, y, 'cubic')

xlocal = np.linspace(0, 5*np.pi, 200)
ylocal = localinterp(xlocal)

plt.plot(x,y,'ro')
plt.plot(xlocal,ylocal,'b-')


# In[ ]:


interp_types = ['linear','nearest','zero','slinear',
                'quadratic','cubic','previous','next']


# In[ ]:


def plotinterp(kindin='linear'):

    localinterp = interpolate.interp1d(x, y, kind=kindin)

    xnew = np.linspace(0, 5*np.pi, 200)
    ynew = localinterp(xnew)

    plt.plot(x,y,'ro')
    plt.plot(xnew,ynew,'b-')
    plt.show()
    
ipywidgets.interact(plotinterp,kindin=interp_types)


# ## Higher dimensions, and irregularly spaced points

# In[ ]:


def f(x,y):
    return np.sin(x) + np.cos(y)


# In[ ]:


x = np.linspace(0,4*np.pi,50)
y = np.linspace(0,4*np.pi,50)
X,Y = np.meshgrid(x,y)
Z = f(X,Y)

fig,ax = plt.subplots(figsize=(5,5))
plt.contourf(X,Y,Z)


# In[ ]:


# "measured data"
xdata = np.random.uniform(0,4*np.pi,100)
ydata = np.random.uniform(0,4*np.pi,100)
zdata = f(xdata,ydata)


# In[ ]:


fig,ax = plt.subplots(figsize=(5,5))
plt.scatter(xdata, ydata, c=zdata)


# In[ ]:


x = np.linspace(0,4*np.pi,50)
y = np.linspace(0,4*np.pi,50)
X,Y = np.meshgrid(x,y)
Z = f(X,Y)

fig,ax = plt.subplots(figsize=(5,5))
plt.contourf(X,Y,Z)
plt.scatter(xdata, ydata, c=zdata, edgecolors='k')


# In[ ]:


Zinterp = interpolate.griddata((xdata, ydata), zdata, (X, Y), method='nearest')


# In[ ]:


Zinterp.shape


# In[ ]:


fig,ax = plt.subplots(figsize=(5,5))
plt.contourf(X,Y,Zinterp)
plt.scatter(xdata, ydata, c=zdata, edgecolors='k')


# In[ ]:


Zinterp = interpolate.griddata((xdata, ydata), zdata, (X, Y), method='linear')
fig,ax = plt.subplots(figsize=(5,5))
plt.contourf(X,Y,Zinterp)
plt.scatter(xdata, ydata, c=zdata, edgecolors='k')


# In[ ]:


Zinterp = interpolate.griddata((xdata, ydata), zdata, (X, Y), method='cubic')
fig,ax = plt.subplots(figsize=(5,5))
plt.contourf(X,Y,Zinterp)
plt.scatter(xdata, ydata, c=zdata, edgecolors='k')


# In[ ]:


Zinterp[0][0]


# In[ ]:




