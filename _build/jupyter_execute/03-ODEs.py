#!/usr/bin/env python
# coding: utf-8

# In solving ODEs, there are several options:
# 
# * Wetware method --> put your hand on a pencil and write the solution
# * Symbolic method --> use symbolic manipulation with software to get an analytical answer
# * Numerical method --> use approximate methods to estimate the solution without producing an analytical answer

# In[ ]:


import numpy as np
import sympy
from scipy import integrate
import matplotlib.pyplot as plt
import ipywidgets


# In[ ]:


sympy.init_printing()


# # Harmonic oscillator

# ## Symbolic manipulation (with SymPy)

# In[ ]:


t = sympy.Symbol("t")
omega0 = sympy.Symbol("omega0")
x = sympy.Function('x')


# In[ ]:


ode = x(t).diff(t, 2) + omega0**2 * x(t)


# In[ ]:


ode


# In[ ]:


ode_sol = sympy.dsolve(ode)


# In[ ]:


ode_sol


# In[ ]:


ode_sol.rhs


# In[ ]:


# initial conditions
ics = {x(0): 2, x(t).diff(t).subs(t, 0): 3}


# In[ ]:


ics


# In[ ]:


ode_sol = sympy.dsolve(ode,ics=ics)


# In[ ]:


ode_sol


# In[ ]:


ode_sol.rewrite(sympy.cos).simplify()


# In[ ]:


ics


# In[ ]:


ode_sol = sympy.dsolve(ode)


# In[ ]:


ode_sol


# In[ ]:


ode_sol.free_symbols


# In[ ]:


ode_sol.free_symbols - {omega0}


# In[ ]:


(ode_sol.lhs.diff(t,0) - ode_sol.rhs.diff(t,0)).subs(t,0)


# In[ ]:


(ode_sol.lhs.diff(t,0) - ode_sol.rhs.diff(t,0)).subs(t,0).subs(ics)


# In[ ]:


(ode_sol.lhs.diff(t,1) - ode_sol.rhs.diff(t,1)).subs(t,0)


# In[ ]:


(ode_sol.lhs.diff(t,1) - ode_sol.rhs.diff(t,1)).subs(t,0).subs(ics)


# In[ ]:


eqs = [(ode_sol.lhs.diff(t, n) - ode_sol.rhs.diff(t, n)).subs(t, 0).subs(ics)
       for n in range(len(ics))]


# In[ ]:


eqs


# In[ ]:


ode_sol.free_symbols - {omega0}


# In[ ]:


sympy.solve(eqs, ode_sol.free_symbols - set([omega0]))


# In[ ]:


sol_params = sympy.solve(eqs, ode_sol.free_symbols - set([omega0]))


# In[ ]:


x_t_sol = ode_sol.subs(sol_params)


# In[ ]:


x_t_sol


# In[ ]:


x_t_sol.rewrite(sympy.cos).simplify()


# Let's use lambdify to plot:  "The primary purpose of this function [lambdify] is to provide a bridge from SymPy expressions to numerical libraries such as NumPy, SciPy, NumExpr, mpmath,
# and tensorflow."

# In[ ]:


square = sympy.lambdify(t, t**2)


# In[ ]:


square(6)


# In[ ]:


np.linspace(0,1,10)


# In[ ]:


square(np.linspace(0,1,10))


# In[ ]:


get_ipython().run_line_magic('pinfo', 'sympy.lambdify')


# In[ ]:


expr_func = sympy.lambdify(t, x_t_sol.rhs.subs(omega0,1), 'numpy')


# In[ ]:


xvalues = np.linspace(0, 10, 30)


# In[ ]:


expr_func(xvalues)


# In[ ]:


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

# In[ ]:


def dxdt(X,t=0):
    omega0 = 1
    return np.array([ X[1] , -omega0**2 * X[0] ])


# In[ ]:


xvalues = np.linspace(0, 10, 30)
scipysoln = integrate.odeint(dxdt, [2,3], xvalues)


# In[ ]:


scipysoln


# In[ ]:


plt.plot(xvalues,scipysoln[:,0],'ro',
         xvalues,expr_func(xvalues));


# # Damped harmonic oscillator

# In[ ]:


t = sympy.Symbol("t")
omega0 = sympy.Symbol("omega0")
gamma = sympy.Symbol("gamma")
x = sympy.Function('x')


# In[ ]:


x = sympy.Function("x")


# In[ ]:


ode = x(t).diff(t, 2) + 2 * gamma * omega0 * x(t).diff(t) + omega0**2 * x(t)


# In[ ]:


ode


# In[ ]:


ode_sol = sympy.dsolve(ode)


# In[ ]:


ode_sol


# In[ ]:


ics = {x(0): 1, x(t).diff(t).subs(t, 0): 0}


# In[ ]:


ics


# In[ ]:


ode_sol = sympy.dsolve(ode,ics=ics)


# In[ ]:


ode_sol


# In[ ]:


# sol = ode_sol
# known_params = [omega0, gamma]
# x = t
# free_params = sol.free_symbols - set(known_params)
# eqs = [(sol.lhs.diff(x, n) - sol.rhs.diff(x, n)).subs(x, 0).subs(ics)
#        for n in range(len(ics))]
# sol_params = sympy.solve(eqs, free_params)
# x_t_sol = sol.subs(sol_params)


# In[ ]:


# eqs


# In[ ]:


# free_params


# In[ ]:


# type(free_params)


# In[ ]:


# ode_sol


# In[ ]:


# x_t_sol


# In[ ]:


# x_t_critical = sympy.limit(x_t_sol.rhs, gamma, 1)


# In[ ]:


# x_t_critical


# In[ ]:


def plotDampedOsc(omega0in=15,gammain=0.1):
#     xsoln = sympy.lambdify(t, x_t_sol.rhs.subs({omega0: omega0in, gamma: gammain}), 'numpy')
    xsoln = sympy.lambdify(t, ode_sol.rhs.subs({omega0: omega0in, gamma: gammain}), 'numpy')
    taxis = np.linspace(0, 3, 100)
    plt.plot(taxis,xsoln(taxis).real)
    plt.ylim(-1.5,1.5)


# In[ ]:


plotDampedOsc(2*np.pi, 0.2)


# In[ ]:


ipywidgets.interact(plotDampedOsc,omega0in=(1,20),gammain=(0.02,1.5,0.02))


# In[ ]:


sympy.limit(x_t_sol.rhs, gamma, 1)


# # Lorenz Equations
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

# In[ ]:


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


# In[ ]:


time_points


# In[ ]:


def lorenz(state, t):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


# In[ ]:


lorenzsoln = integrate.odeint(lorenz, initial_state, time_points)


# In[ ]:


fig = plt.figure(figsize=(12, 9))
ax = fig.gca(projection='3d')
x = lorenzsoln[:, 0]
y = lorenzsoln[:, 1]
z = lorenzsoln[:, 2]
ax.plot(x, y, z, color='g', alpha=0.7, linewidth=0.7)


# In[ ]:


import ipywidgets


# In[ ]:


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




