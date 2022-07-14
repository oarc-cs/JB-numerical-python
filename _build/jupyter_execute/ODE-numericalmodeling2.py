#!/usr/bin/env python
# coding: utf-8

# # Ex. Solving an ODE with Numerical Modeling

# In[1]:


import numpy as np
import sympy
from scipy import integrate
import matplotlib.pyplot as plt
import ipywidgets


# In[2]:


sympy.init_printing()


# ## Harmonic oscillator

# ### Numerical manipulation (with SciPy)

# Equation:
#     
# $$
# \left(\frac{d^2}{dt^2} + \omega_0^2\right) x(t) = 0
# $$
# 
# write this as two first-order equations
# 
# $$
# x'(t) = y(t)
# $$
# $$
# y'(t) = \omega_0^2 x(t)
# $$

# In[3]:


def dxdt(X,t=0):
    omega0 = 1
    return np.array([ X[1] , -omega0**2 * X[0] ])


# In[4]:


xvalues = np.linspace(0, 10, 30)
scipysoln = integrate.odeint(dxdt, [2,3], xvalues)


# In[5]:


scipysoln


# In[6]:


plt.plot(xvalues,scipysoln[:,0],'ro',
         xvalues,expr_func(xvalues));


# ## Damped Harmonic Oscillator

# In[44]:


t = sympy.Symbol("t")
omega0 = sympy.Symbol("omega0")
gamma = sympy.Symbol("gamma")
x = sympy.Function('x')


# In[45]:


x = sympy.Function("x")


# In[46]:


ode = x(t).diff(t, 2) + 2 * gamma * omega0 * x(t).diff(t) + omega0**2 * x(t)


# In[47]:


ode


# In[48]:


ode_sol = sympy.dsolve(ode)


# In[49]:


ode_sol


# In[50]:


ics = {x(0): 1, x(t).diff(t).subs(t, 0): 0}


# In[51]:


ics


# In[52]:


ode_sol = sympy.dsolve(ode,ics=ics)


# In[53]:


ode_sol


# In[54]:


# sol = ode_sol
# known_params = [omega0, gamma]
# x = t
# free_params = sol.free_symbols - set(known_params)
# eqs = [(sol.lhs.diff(x, n) - sol.rhs.diff(x, n)).subs(x, 0).subs(ics)
#        for n in range(len(ics))]
# sol_params = sympy.solve(eqs, free_params)
# x_t_sol = sol.subs(sol_params)


# In[55]:


# eqs


# In[56]:


# free_params


# In[57]:


# type(free_params)


# In[58]:


# ode_sol


# In[59]:


# x_t_sol


# In[60]:


# x_t_critical = sympy.limit(x_t_sol.rhs, gamma, 1)


# In[61]:


# x_t_critical


# In[62]:


def plotDampedOsc(omega0in=15,gammain=0.1):
#     xsoln = sympy.lambdify(t, x_t_sol.rhs.subs({omega0: omega0in, gamma: gammain}), 'numpy')
    xsoln = sympy.lambdify(t, ode_sol.rhs.subs({omega0: omega0in, gamma: gammain}), 'numpy')
    taxis = np.linspace(0, 3, 100)
    plt.plot(taxis,xsoln(taxis).real)
    plt.ylim(-1.5,1.5)


# In[63]:


plotDampedOsc(2*np.pi, 0.2)


# In[64]:


ipywidgets.interact(plotDampedOsc,omega0in=(1,20),gammain=(0.02,1.5,0.02))


# In[65]:


sympy.limit(x_t_sol.rhs, gamma, 1)


# ## Lorenz Equations
# 
# $$
# x'(t) = \sigma(y - x)
# $$
# $$
# y'(t) = x(\rho - z) - y 
# $$
# $$
# z'(t) = x y - \beta z
# $$

# In[66]:


# define the initial system state (aka x, y, z positions in space)
initial_state = [0.1, 0, 0]

# define the system parameters sigma, rho, and beta
sigma = 10.
rho   = 28.
beta  = 8./3.

# define the time points to solve for, evenly spaced between the start and end times
start_time = 1
end_time = 60
interval = 100
time_points = np.linspace(start_time, end_time, end_time * interval)


# In[67]:


time_points


# In[68]:


def lorenz(state, t):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


# In[69]:


lorenzsoln = integrate.odeint(lorenz, initial_state, time_points)


# In[70]:


fig = plt.figure(figsize=(12, 9))
ax = fig.gca(projection='3d')
x = lorenzsoln[:, 0]
y = lorenzsoln[:, 1]
z = lorenzsoln[:, 2]
ax.plot(x, y, z, color='g', alpha=0.7, linewidth=0.7)


# In[71]:


import ipywidgets


# In[72]:


def plotlorenz(end=1):
    start_time = 1
    end_time = end
    interval = 100
    time_points = np.linspace(start_time, end_time, end_time * interval)
    lorenzsoln = integrate.odeint(lorenz, initial_state, time_points)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.gca(projection='3d')
    x = lorenzsoln[:, 0]
    y = lorenzsoln[:, 1]
    z = lorenzsoln[:, 2]
    ax.plot(x, y, z, color='g', alpha=0.7, linewidth=0.7)

ipywidgets.interact(plotlorenz,end=(1,60))


# In[ ]:





# In[ ]:





# In[ ]:




