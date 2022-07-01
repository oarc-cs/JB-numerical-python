#!/usr/bin/env python
# coding: utf-8

# ## Numerical Computing with Python, Part 3:   Numerical Modeling  <a class="tocSkip">
#     
# To start, let's do a bit of symbolic math to get the gears churning...

# In[ ]:


import sympy
sympy.init_printing()


# In[ ]:


x = sympy.Symbol('x')


# In[ ]:


x


# In[ ]:


x**2


# In[ ]:


expr = x**4 + x**3 + x**2 + x + 1


# In[ ]:


expr


# In[ ]:


expr.diff(x)


# In[ ]:


expr.diff(x,3)


# In[ ]:


expr.diff(x,3).expand()


# In[ ]:


expr.integrate(x)


# In[ ]:


expr.integrate((x,0,3))


# In[ ]:


sympy.integrate(expr,x)


# In[ ]:


sympy.integrate(expr,(x,0,3))


# In[ ]:


sympy.cos(x)


# In[ ]:


sympy.cos(x).series(x,n=10)


# In[ ]:


sympy.cos(x).integrate(x)


# In[ ]:


sympy.tan(x)


# In[ ]:


sympy.tan(x).integrate(x)


# In[ ]:


n = sympy.symbols("n")


# In[ ]:


from sympy import oo


# In[ ]:


x = sympy.Sum(1/(n**2), (n, 1, oo))


# In[ ]:


x


# In[ ]:


x = sympy.Symbol('x')


# In[ ]:


a = sympy.Sum(x**n/(sympy.factorial(n)), (n, 0, oo))


# In[ ]:


a


# In[ ]:


a.doit()


# In[ ]:


sympy.Matrix([1,2])


# In[ ]:


import numpy as np


# In[ ]:


b = np.array([[1,2],[3,4]])


# In[ ]:


b


# In[ ]:


m = sympy.Matrix(b)


# In[ ]:


m


# In[ ]:


type(m)

