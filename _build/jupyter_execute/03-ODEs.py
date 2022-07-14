#!/usr/bin/env python
# coding: utf-8

# # Ordinary Differential Equations
# In solving ODEs, there are several options:
# 
# * Wetware method --> put your hand on a pencil and write the solution
# * Symbolic method --> use symbolic manipulation with software to get an analytical answer
# * Numerical method --> use approximate methods to estimate the solution without producing an analytical answer

# # Ordinary Differential Equations
# In solving ODEs, there are several options:
# 
# * Wetware method --> put your hand on a pencil and write the solution
# * Symbolic method --> use symbolic manipulation with software to get an analytical answer
# * Numerical method --> use approximate methods to estimate the solution without producing an analytical answer

# In[1]:


import numpy as np
import sympy
from scipy import integrate
import matplotlib.pyplot as plt
import ipywidgets


# In[2]:


sympy.init_printing()


# ## Harmonic oscillator

# ## Symbolic manipulation (with SymPy)

# In[3]:


t = sympy.Symbol("t")
omega0 = sympy.Symbol("omega0")
x = sympy.Function('x')


# In[4]:


ode = x(t).diff(t, 2) + omega0**2 * x(t)


# In[5]:


ode


# In[6]:


ode_sol = sympy.dsolve(ode)


# In[7]:


ode_sol


# In[8]:


ode_sol.rhs


# In[9]:


# initial conditions
ics = {x(0): 2, x(t).diff(t).subs(t, 0): 3}


# In[10]:


ics


# In[11]:


ode_sol = sympy.dsolve(ode,ics=ics)


# In[12]:


ode_sol


# In[13]:


ode_sol.rewrite(sympy.cos).simplify()


# In[14]:


ics


# In[15]:


ode_sol = sympy.dsolve(ode)


# In[16]:


ode_sol


# In[17]:


ode_sol.free_symbols


# In[18]:


ode_sol.free_symbols - {omega0}


# In[19]:


(ode_sol.lhs.diff(t,0) - ode_sol.rhs.diff(t,0)).subs(t,0)


# In[20]:


(ode_sol.lhs.diff(t,0) - ode_sol.rhs.diff(t,0)).subs(t,0).subs(ics)


# In[21]:


(ode_sol.lhs.diff(t,1) - ode_sol.rhs.diff(t,1)).subs(t,0)


# In[22]:


(ode_sol.lhs.diff(t,1) - ode_sol.rhs.diff(t,1)).subs(t,0).subs(ics)


# In[23]:


eqs = [(ode_sol.lhs.diff(t, n) - ode_sol.rhs.diff(t, n)).subs(t, 0).subs(ics)
       for n in range(len(ics))]


# In[24]:


eqs


# In[25]:


ode_sol.free_symbols - {omega0}


# In[26]:


sympy.solve(eqs, ode_sol.free_symbols - set([omega0]))


# In[27]:


sol_params = sympy.solve(eqs, ode_sol.free_symbols - set([omega0]))


# In[28]:


x_t_sol = ode_sol.subs(sol_params)


# In[29]:


x_t_sol


# In[30]:


x_t_sol.rewrite(sympy.cos).simplify()


# Let's use lambdify to plot:  "The primary purpose of this function [lambdify] is to provide a bridge from SymPy expressions to numerical libraries such as NumPy, SciPy, NumExpr, mpmath,
# and tensorflow."

# In[31]:


square = sympy.lambdify(t, t**2)


# In[32]:


square(6)


# In[33]:


np.linspace(0,1,10)


# In[34]:


square(np.linspace(0,1,10))


# In[35]:


get_ipython().run_line_magic('pinfo', 'sympy.lambdify')


# In[36]:


expr_func = sympy.lambdify(t, x_t_sol.rhs.subs(omega0,1), 'numpy')


# In[37]:


xvalues = np.linspace(0, 10, 30)


# In[38]:


expr_func(xvalues)


# In[39]:


plt.plot(xvalues,expr_func(xvalues));


# ## Numerical manipulation (with SciPy)

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

# In[40]:


def dxdt(X,t=0):
    omega0 = 1
    return np.array([ X[1] , -omega0**2 * X[0] ])


# In[41]:


xvalues = np.linspace(0, 10, 30)
scipysoln = integrate.odeint(dxdt, [2,3], xvalues)


# In[42]:


scipysoln


# In[43]:


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




