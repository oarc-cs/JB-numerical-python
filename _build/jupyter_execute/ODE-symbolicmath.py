#!/usr/bin/env python
# coding: utf-8

# # Ex. Solving an ODE with Symbolic Math

# In[1]:


import numpy as np
import sympy
from scipy import integrate
import matplotlib.pyplot as plt
import ipywidgets


# In[2]:


sympy.init_printing()


# ## Harmonic Oscillator

# ### Symbolic Manipulation (with SymPy)

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


# Let's use lambdify to plot: "The primary purpose of this function [lambdify] is to provide a bridge from SymPy expressions to numerical libraries such as NumPy, SciPy, NumExpr, mpmath, and tensorflow."

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


lt.plot(xvalues,expr_func(xvalues));


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




