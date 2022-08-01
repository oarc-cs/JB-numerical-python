#!/usr/bin/env python
# coding: utf-8

# # Matplotlib

# <img src="./images/matplotlib_image.webp" width=600>
# Anatomy of a Figure, https://matplotlib.org/3.1.1/gallery/showcase/anatomy.html

# In[1]:


import matplotlib.pyplot as plt


# A list of numbers into a graphic

# In[2]:


x = [1, 2, 3, 4]
y = [10, 11, 12, 13]


# In[3]:


plt.plot(x,y)


# In[4]:


plt.plot(x,y,'ro')


# In[5]:


plt.bar(x,y)


# In[6]:


plt.scatter(x,y,color='r')


# Just so you see it, there are two main ways to create plots in matplotlib
# 
# 1. Use matplotlib.pyplot (which here is aliased as plt) -- this is the higher-level and easier to use module
# 2. Use figure and axes objects (objects in the object-oriented programming sense) to manipulate the graphical object you see

# In[7]:


# the object way
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x,y)


# In[8]:


fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1, 1, 1)

ax.plot(x,y)


# In[9]:


fig,ax = plt.subplots(1,2,figsize=(8,4))

x = [1,2,3,4]
y1 = [1,2,3,4]
y2 = [1,4,9,16]

ax[0].plot(x,y1)
ax[1].plot(x,y2,color='blue', linestyle='--', linewidth=2)

plt.show()


# Let's stick with plt for the moment (and later we'll migrate to pandas)

# In[10]:


x = np.linspace(0,2*np.pi,100)
y = np.cos(x)
plt.plot(x,y)


# In[ ]:


x = np.linspace(0,2*np.pi,100)
y = np.cos(x)
plt.figure(figsize=(8,5))
plt.plot(x,y)


# In[ ]:


x = np.linspace(0,2*np.pi,100)
y = np.cos(x)
plt.figure(figsize=(8,5))
plt.plot(x,y)
plt.xlim([0,2*np.pi])
plt.xlabel('position')
plt.ylabel('amplitude')
plt.show()


# In[ ]:


# figure call first
plt.figure(figsize=(8,5))

# specification for scatter plot
x2 = np.linspace(0,2*np.pi,10)
y2 = np.cos(x2)
plt.scatter(x2,y2,color='black')

# specification for line plot -- more points so it looks smooth
x1 = np.linspace(0,2*np.pi,100)
y1 = np.cos(x)
plt.plot(x,y)

# specification for axes, labels, etc.
plt.xlim([0,2*np.pi])
plt.xlabel('position')
plt.ylabel('amplitude')

plt.show()

