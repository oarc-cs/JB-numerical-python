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


# ## Harmonic Oscillator

# ### Numerical Manipulation (with SciPy)

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
