#!/usr/bin/env python
# coding: utf-8

# # NumPy and SciPy 

# ## Using NumPy to Look at Series of Data

# In[1]:


# set these as constants
p = 1000
y = 1
c = 100


# In[2]:


import benpy


# In[3]:


# %load benpy.py
def compound_calculator(principal,rate,year,contribution):
    '''
    compound_calculator(p,r,y,c) calculates the value at year y of an investment
    p = principal
    r = interest rate (percent value)
    y = year
    c = contribution at end of each year
    '''

    p = principal
    r = rate/100
    y = year
    c = contribution
    balance = p*(1 + r)**y + c*( ((1 + r)**(y+1) - (1 + r)) / r )
    return balance


# In[4]:


rates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# Calculate the balance for each of the given rates 

# In[5]:


balance = []
for i in rates:
    r = i
    balance.append(benpy.compound_calculator(p,r,y,c))
print(balance)


# This is what we might like to do:
# `balance = benpy.compound_calculator(p,rates,y,c)
# * Has the list of rates directly in the function call

# In[6]:


import numpy as np


# In[7]:


ratesnp = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


# In[8]:


ratesnp


# In[9]:


balancenp = benpy.compound_calculator(p,ratesnp,y,c)


# In[10]:


balancenp


# ### We're going to use a bit of ipython "magics" to look at timings.... this won't work in scripts, but works well here in the Jupyter notebook
# * %%timeit times everything in the given cell

# In[11]:


get_ipython().run_cell_magic('timeit', '', 'balance = []\nfor i in rates:\n    r = i\n    balance.append(benpy.compound_calculator(p,r,y,c))')


# In[12]:


get_ipython().run_cell_magic('timeit', '', 'balancenp = benpy.compound_calculator(p,ratesnp,y,c)')


# In[13]:


ratesnp = np.arange(1,11)


# In[14]:


ratesnp


# In[15]:


get_ipython().run_cell_magic('timeit', '', 'balance = []\nfor i in range(1,10000):\n    r = i\n    balance.append(benpy.compound_calculator(p,r,y,c))')


# In[16]:


get_ipython().run_cell_magic('timeit', '', 'ratesnp = np.arange(1,10000)\nbalancenp = benpy.compound_calculator(p,ratesnp,y,c)')


# ### Basic Operations with N-dimensional Arrays

# In[17]:


ratesnp


# In[18]:


a = np.array([[1,2],[3,4]])
b = np.array([[-4,-3],[-2,-1]])


# In[19]:


a


# In[20]:


b


# In[21]:


a+b


# In[22]:


a-b


# In[23]:


a/b


# In[24]:


a*b


# In[25]:


np.matmul(a,b) # matrix multiplication


# In[26]:


# array attributes
print(a.ndim)
print(a.shape)
print(a.size)
print(a.dtype)


# In[27]:


a.T


# #### Indexing and Slicing

# In[28]:


a[0]


# In[29]:


a[0:2]


# In[30]:


a[1:4]


# In[31]:


a[2:4]


# In[32]:


a


# In[33]:


a[0:1,0]


# In[34]:


a[:,0]


# In[35]:


a[1,:]


# In[36]:


a > 2


# In[37]:


a[a > 2] #returns all elements that are greater than 2


# In[38]:


a[a % 2 == 0]


# #### Creating Some Arrays

# In[39]:


np.arange(10)


# In[40]:


np.arange(1,11,0.5)


# In[41]:


np.arange(-1,1,0.2)


# In[42]:


np.linspace(-1,1,11)


# In[43]:


np.linspace(0,2*np.pi,100)


# In[44]:


x = np.linspace(0,2*np.pi,100)
y = np.cos(x)


# In[45]:


y


# Let's plot for fun (cosine plot).... briefly use matplotlib

# In[46]:


import matplotlib.pyplot as plt


# In[47]:


plt.plot(x,y,'ro')


# Reshape Arrays

# In[48]:


a2 = np.arange(10).reshape((2,5)) 


# In[49]:


a2


# In[50]:


a2.reshape(10) 


# In[51]:


a2.reshape((3,4))


# In[ ]:


a2.reshape((10,1)) #ten rows and one column


# ### Broadcasting

# In[26]:


a2 = a2.reshape((10,1))


# In[27]:


a2


# In[28]:


b2 = np.array([1,2,3])


# In[29]:


b2


# In[30]:


a2 + b2


# Broadcasting:  numpy will compare array shapes and consider respective dimensions to be compatible if:
# 1. they are equal
# 2. one of them is 1

# In[ ]:


a3 = np.array([[3,4,5],[6,7,8]])
b3 = np.array([[1,2,3]])


# In[ ]:


a3 * b3


# Another (intuitive) example:  adding or multiplying a matrix by a scalar
# * Every dimension of a scalar is one

# In[ ]:


2 * a3 + 1


# ### Operations Along Axes

# In[31]:


a


# In[35]:


a.sum() # sum of all elements in an array


# In[ ]:


a.sum(axis=0) # sum of every column


# In[ ]:


a.sum(axis=1) # sum of every row


# In[33]:


a.cumsum() # cumulative sum


# In[34]:


a.cumsum(axis=1) # cumulative sum of each row


# In[ ]:


a.min() # min of entire array


# In[ ]:


a.min(axis=0)


# In[ ]:


a.max() # max of entire array


# In[ ]:


a.max(axis=1)


# ### Example: Estimating $\pi$

# ![image.png](attachment:image.png)

# The fraction of sample points that make it into the circle is:
# 
# $$\frac{N_{inside}}{N_{total}} = \frac{\pi r^2}{4 r^2}$$
# 
# so we can use our sample to calculate $\pi$ via:
# 
# $$\pi = 4 \frac{N_{inside}}{N_{total}}$$

# In[39]:


np.random.uniform(0,1) # generates a random point between 0 and 1


# In[40]:


x = np.random.uniform(0,1,1000) # creates an array of 1000 random points between 0 and 1
y = np.random.uniform(0,1,1000)
in_circle = (((x-0.5)**2 + (y-0.5)**2) < 0.5**2) 


# In[38]:


in_circle # returns which points in the array fall in the circle


# In[41]:


np.unique(in_circle, return_counts=True) # returns the number of false values and true values


# In[42]:


in_unique, in_counts = np.unique(in_circle, return_counts=True)


# In[43]:


in_counts


# In[44]:


4 * in_counts[1] / 1000 # estimate of pi


# In[45]:


# the above steps put into one function
def pi_estimate(nums = 1000):
    x = np.random.uniform(0,1,nums)
    y = np.random.uniform(0,1,nums)
    in_circle = (((x-0.5)**2 + (y-0.5)**2) < 0.5**2)
    in_unique, in_counts = np.unique(in_circle, return_counts=True)
    estimated_pi = 4 * in_counts[1] / nums
    print('pi = '+str(estimated_pi))
    return x,y


# In[46]:


pi_estimate(100)


# In[47]:


fig,ax = plt.subplots(figsize=(5,5))
x,y = pi_estimate(100)
plt.plot(x,y,'ro')
plt.axis([0, 1, 0, 1])
circle1 = plt.Circle((0.5, 0.5), 0.5)
plt.gca().add_patch(circle1)
plt.show();


# ## SciPy
# 
# * SciPy is the core package for scientific routines in Python
# * it operates efficiently on numpy arrays and the two are intended to work together
# * Many sub-modules are available:
#   * *scipy.cluster* - Vector quantization / Kmeans
#   * *scipy.constants* - Physical and mathematical constants
#   * *scipy.fftpack* - Fourier transform
#   * *scipy.integrate* - Integration routines
#   * *scipy.interpolate* - Interpolation
#   * *scipy.io* - Data input and output
#   * *scipy.linalg* - Linear algebra routines
#   * *scipy.ndimage* - n-dimensional image package
#   * *scipy.odr* - Orthogonal distance regression
#   * *scipy.optimize* - Optimization
#   * *scipy.signal* - Signal processing
#   * *scipy.sparse* - Sparse matrices
#   * *scipy.spatial* - Spatial data structures and algorithms
#   * *scipy.special* - Any special mathematical functions
#   * *scipy.stats* - Statistics
# * I will show brief examples here and discuss some more scipy material next week.
# 
# [Acknowledgement goes to examples from the scipy docs]

# ### Interpolation

# In[50]:


x = np.linspace(0,2*np.pi,10)
noise = (np.random.random(10)*2 - 1) * 0.1
y = np.cos(x) + noise
plt.plot(x,y,'ro')


# In[51]:


from scipy.interpolate import interp1d


# In[52]:


# linear interpolation 
linear_interp = interp1d(x, y)

xlin = np.linspace(0, 2*np.pi, 100)
ylin = linear_interp(xlin)

plt.plot(x,y,'ro')
plt.plot(xlin,ylin,'b-')


# In[53]:


# cubic interpolation
cubic_interp = interp1d(x, y, kind='cubic')

xcub = np.linspace(0, 2*np.pi, 100)
ycub = cubic_interp(xcub)

plt.plot(x,y,'ro')
plt.plot(xcub,ycub,'b-')


# ### Images

# In[54]:


import imageio
from scipy import misc
from scipy import ndimage
import numpy as np


# In[55]:


parrot = imageio.imread('parrot.jpeg')


# In[56]:


plt.imshow(parrot)


# In[57]:


noisy_parrot = np.copy(parrot).astype(np.float)
noisy_parrot += parrot.std() * 0.1 * np.random.standard_normal(parrot.shape)
# noisy_parrot = ndimage.gaussian_filter(parrot, sigma=5)


# In[58]:


plt.imshow(noisy_parrot.astype('uint8'))


# In[59]:


noisy_parrot.shape


# In[60]:


plt.imshow(noisy_parrot[400:,500:,].astype('uint8')) # zoom in


# In[65]:


median_parrot = ndimage.median_filter(noisy_parrot, size=5)


# In[62]:


plt.imshow(median_parrot[400:,500:,].astype('uint8'))


# In[63]:


rotated_parrot = ndimage.rotate(noisy_parrot, -90)


# In[64]:


plt.imshow(rotated_parrot[500:,:400,].astype('uint8')) # rotate image


# <img src="parrot.jpeg” alt="parrot” width="200">

# In[ ]:




