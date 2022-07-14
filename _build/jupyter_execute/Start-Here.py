#!/usr/bin/env python
# coding: utf-8

# # Python Basics

# ![image.png](attachment:image.png)

# Python was developed by Guido van Rossum and released in 1991.
# 
# Its core philosophy includes aphorisms like:
# * Beautiful is better than ugly.
# * Explicit is better than implicit.
# * Simple is better than complex.
# * Complex is better than complicated.
# * Readability counts.
# 

# "Python is an interpreted, object-oriented, high-level programming language with dynamic semantics."
# 
# * A general purpose programming language
# * Interpreted -- you don't have to compile Python code in order to run it.
# * The simplicity and readability can increase productivity
# * It can be very easy to learn, but it's also a powerful language with many libraries that enhance its ability to efficiently tackle a wide range of computational problems.
# * Very well suited for interactive work and quick prototyping, while being powerful enough to write large applications in.

# Another interesting tidbit -- "Python" comes from Monty Python's Flying Circus, rather than the snake.

# ## Various Ways to Run Python code

# * Interactively at a prompt
# * With a file that contains the code you want
# * Inside of an interactive development environment

# Run each cell using the live code feature, which can be found in the upper right corner
# * each cell provides examples to further undertsand some fundamentals of Python

# ## Calculating

# Example: compound interest calculator with annual contributions
# 
# * p = principal
# * r = annual interest rate in percent
# * y = year of the balance
# * c = annual contribution (made at the start of the year)
# 
# $$\text{Balance}(y) = p(1 + r)^y + c\left[\frac{(1 + r)^{y+1} - (1 + r)}{r} \right]$$
# 
# * a variable is denoted with a "=" sign

# In[1]:


y1 = 1
p1 = 1000
r1 = 0.05
c1 = 100
p1*(1 + r1)**y1 + c1*( ((1 + r1)**(y1 + 1) - (1 + r1)) / r1 )


# In[2]:


y2 = 45
p2 = 1
r2 = 0.05
c2 = 6500
p2*(1 + r2)**y2 + c2*( ((1 + r2)**(y2 + 1) - (1 + r2)) / r2 )


# ## Strings

# A string is a sequence of characters surrounded by quote marks

# In[3]:


parrotreturn = "This parrot is no more! It has ceased to be!"


# Index elements of the string with brackets 

# In[4]:


parrotreturn[1]


# In[5]:


parrotreturn[0]


# Do note:  Python numbering starts at 0, not 1

# In[6]:


parrotreturn[0:3]


# In[7]:


parrotreturn[0:4]


# Slicing includes the first number but not the last

# In[8]:


parrotreturn.index('!')


# What exactly is ".index"? -- index is a *method* and the "." notifies Python to call the method associated with parrotreturn.
# * returns the index of the given element

# In[9]:


# Use parrotreturn.index('!') and slicing to print out only the first sentence of the string


# In[10]:


print(parrotreturn)


# In[11]:


parrotreturn = '"This parrot is no more! It has ceased to be!"'


# In[12]:


print(parrotreturn)


# In[13]:


parrotreturn = "\"This parrot is no more! It has ceased to be!\""


# In[14]:


print(parrotreturn)


# In[15]:


parrotreturn = "\"This parrot is no more!\"\n\"It has ceased to be!\""


# In[16]:


print(parrotreturn)


# Strings can also be concatenated and expanded with math operators

# In[17]:


"spam"


# In[18]:


"spam"*3


# In[19]:


'spam'*3 + ' and ham and eggs'


# In[20]:


return3 = "It's expired and gone to meet its maker!"


# In[21]:


parrotreturn + return3


# In[22]:


print(parrotreturn + return3)


# In[23]:


# clean up the quote so that the formatting is consistent


# In[24]:


# put the entire string into one variable
# and use the bracket syntax with slicing to print only the middle sentence


# ## Lists

# A list is a collection of items which is bounded by square brackets

# In[25]:


holyhandgrenade = [1, 2, 5]


# Index elements of the list
# * ex. index #2 = 5

# In[26]:


holyhandgrenade[1]


# In[27]:


holyhandgrenade[0]


# In[28]:


holyhandgrenade[3]


# In[ ]:


holyhandgrenade[-1]


# In[ ]:


holyhandgrenade[-3]


# In[ ]:


holyhandgrenade[-4]


# In[ ]:


holyhandgrenade[-3:-1]


# In[ ]:


holyhandgrenade[-3:0]


# In[ ]:


holyhandgrenade[-3:]


# In[ ]:


holyhandgrenade[:]


# In[ ]:


holyhandgrenade[2]


# Replace an element of the list at a given index

# In[ ]:


holyhandgrenade[2] = 3


# In[ ]:


holyhandgrenade = ['one','two','five']


# In[ ]:


holyhandgrenade


# In[ ]:


holyhandgrenade.sort()


# In[ ]:


holyhandgrenade


# In[ ]:


holyhandgrenade.reverse()


# In[ ]:


holyhandgrenade


# In[ ]:


help(holyhandgrenade)


# In[ ]:


holyhandgrenade.append('three')


# In[ ]:


holyhandgrenade


# In[ ]:


print(holyhandgrenade.__doc__)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'holyhandgrenade')


# In[ ]:


type(holyhandgrenade)


# In[ ]:


# make your own list with 5+ elements


# In[ ]:


# try out the "pop" method to see what it does, then try it with a value between the parentheses


# ## Tuples, Sets, and Dictionaries

# ### Tuples

# In[ ]:


riddleanswers = ('Lancelot', 'Holy Grail', 'blue')


# In[ ]:


riddleanswers[2]


# In[ ]:


riddleanswers[2] = 'green'


# In[ ]:


riddleanswers = ('Lancelot', 43, ['x','y','z'])


# In[ ]:


riddleanswers[2]


# ### Sets

# In[ ]:


riddleanswers = {'Lancelot', 'Holy Grail', 'blue'}


# In[ ]:


riddleanswers[2]


# In[ ]:


'blue' in riddleanswers


# In[ ]:


riddleanswers


# In[ ]:


riddleanswers.add('green')


# In[ ]:


riddleanswers


# In[ ]:


riddleanswers.add('blue')


# In[ ]:


riddleanswers


# ### Dictionary

# dictionary = {'key': 'value', 'key':'value', ... }

# In[ ]:


riddleanswers = {'name':'Lancelot', 'quest':'Holy Grail', 'favorite colour':'blue'}


# In[ ]:


riddleanswers[2]


# In[ ]:


riddleanswers['favorite colour']


# In[ ]:


riddleanswers.keys()


# In[ ]:


riddleanswers.values()


# In[ ]:


# Add a new key/value pair to riddleanswers


# In[ ]:


# Do a quick search on Google and see if you can figure out how to remove "name":"Lancelot" from the dict


# ## Conditionals and Loops

# Mathematical conditions:
# * Equals: `==`
# * Does not equal: `!=`
# * Less than: `<`
# * Less than or equal to: `<=`
# * Greater than: `>`
# * Greater than or equal to: `>=`

# In[ ]:


3 == 4


# In[ ]:


3 != 4


# In[ ]:


3 < 4


# In[ ]:


3 <= 4


# In[ ]:


3 > 4


# In[ ]:


3 >= 4


# ### Executing commands based on a condition:  if, elif, and else

# In[ ]:


if 3 == 4:
    print('3 is equal to 4')


# In[ ]:


if 4 == 4:
    print('This is self-evident')


# Python uses indentation to define blocks of code
# * this is not merely a matter of style in Python
# * it is *very* important for defining blocks of code
# * it is up to you how many spaces you want to use for indentation, as long as you are consistent
# * you can not mix tabs and spaces -- here in Jupyter, the indentation is set automatically to 4 spaces

# In[ ]:


if 3 == 4:
    print('3 is equal to 4')
elif 3 > 4:
    print('3 is greater than 4')


# In[ ]:


if 3 == 4:
    print('3 is equal to 4')
elif 3 < 4:
    print('3 is less than 4')


# In[ ]:


if 3 == 4:
    print('3 is equal to 4')
elif 3 > 4:
    print('3 is greater than 4')
else:
    print('3 is not equal to or greater than 4')


# Combine conditions:
# * and
# * or
# * not

# In[ ]:


(1==1) and (2==2)


# In[ ]:


(1==2) or (2==2)


# In[ ]:


not(1==2)


# In[ ]:


claim = 5
if claim == 1 or claim == 2:
    print('do not throw the grenade yet')
elif claim == 3:
    print('throw the grenade')
elif claim == 5:
    print('Silly Arthur.  3 comes after 2.')
else:
    print('You are too late - kaboom!')


# In[ ]:


# Switch the value of claim above several times and execute
# Make sure you follow the logic of the resulting print statement.


# ### While loops

# The below while loop is an infinite loop

# In[ ]:


disneytrip = 0
while disneytrip == 0:
    print('Are we there yet?')


# In[ ]:


disneytrip = 0
while disneytrip < 10:
    print(disneytrip + ': Are we there yet?')
    disneytrip += 1


# It can be very useful to learn how to decipher these error messages.
# 
# Though we don't often pause to reflect on it, "add" depends on context -- Adding 2 + 2 is different than adding me to your list of workshop instructors

# In[ ]:


disneytrip = 0
while disneytrip < 10:
    print(str(disneytrip) + ': Are we there yet?')
    disneytrip += 1


# ### For loops

# A for loop is used to iterate through a specified range, list, etc. 

# In[ ]:


for disneytrip in range(10): 
    print(str(disneytrip) + ': Are we there yet?')


# In[ ]:


for letter in parrotreturn:
    print(letter)


# In[ ]:


for letter in parrotreturn:
    print(letter, end='')


# In[ ]:


riddleanswers


# In[ ]:


for k in riddleanswers:
    print(k)


# In[ ]:


for k in riddleanswers:
    print(riddleanswers[k])


# In[ ]:


for key, value in riddleanswers.items():
    print(key + ': ' + value)


# In[ ]:


for i in range(3):
    print(i)


# In[ ]:


for i in range(1,4):
    print(i)


# In[ ]:


# Create a for loop that prints the sequence 1, 2, 5


# ## Functions

# Back to our simple example of a compound interest calculator with annual contributions
# 
# * p = principal
# * r = annual interest rate in percent
# * y = year of the balance
# * c = annual contribution (made at the start of the year)
# 
# $$\text{Balance}(y) = p(1 + r)^y + c\left[\frac{(1 + r)^{y+1} - (1 + r)}{r} \right]$$

# Ideally we'd like to plug in a bunch of numbers and see what comes out.
# 
# Functions allow you to collect together a block of code, name it, and run it when called.
# 
# You can pass data into functions and you can get results returned from functions.

# In[ ]:


def f():
    print('Hello World!')


# In[ ]:


f()


# In[ ]:


def f2(a):
    return a*2


# In[ ]:


f2


# In[ ]:


f2(3)


# In[ ]:


f2(a=4)


# In[ ]:


def f(p,r,y,c):
    return p*(1 + r)**y + c*( ((1 + r)**(y+1) - (1 + r)) / r )


# In[ ]:


year = 1
principal = 1000
rate = 5
annual_contribution = 100
f(principal, rate, year, annual_contribution)


# In[ ]:


year = 1
principal = 1000
rate = 5
annual_contribution = 100
f(principal, rate/100, year, annual_contribution)


# In[ ]:


year = 45
principal = 1000
rate = 5
annual_contribution = 6500
f(principal, rate/100, year, annual_contribution)


# In[ ]:


def f2digit(p,r,y,c):
    r = r/100
    return '{:.2f}'.format(p*(1 + r)**y + c*( ((1 + r)**(y+1) - (1 + r)) / r ))


# In[ ]:


year = 45
principal = 1000
rate = 5
annual_contribution = 6500
f2digit(principal, rate, year, annual_contribution)


# In[ ]:


def f2digit(p,r,y,c):
    r = r/100
    amountsaved = '{:.2f}'.format(p*(1 + r)**y + c*( ((1 + r)**(y+1) - (1 + r)) / r ))
    saying = "If you save for " + str(y) + " years, then you'll have $" + amountsaved + " in your retirement."
    return saying


# In[ ]:


year = 45
principal = 1000
rate = 5
annual_contribution = 6500
f2digit(principal, rate, year, annual_contribution)


# What if you want commas in your number?
# 
# * Google is a fantastic reference for python questions
# * Many many common questions have already been asked and answered
# * A quick search may lead you right to the answer you need
# (https://stackoverflow.com/questions/5180365/python-add-comma-into-number-string)

# In[ ]:


def f2digit(p,r,y,c):
    r = r/100
    amountsaved = '{:,.2f}'.format(p*(1 + r)**y + c*( ((1 + r)**(y+1) - (1 + r)) / r ))
    saying = "If you save for " + str(y) + " years, then you'll have $" + amountsaved + " in your retirement."
    return saying


# In[ ]:


year = 45
principal = 1000
rate = 5
annual_contribution = 6500
f2digit(principal, rate, year, annual_contribution)


# If interested check out the documentation for Python.
# * The answers on that stackoverflow page have a link for PEP (Python Enhancement Proposal) and thereby to https://www.python.org/
# * The page also has another link for the official documentation (https://docs.python.org/3/)
# * You may not immediately appreciate all the points on the documentation site, but that is ok.
# * It's intended to be a complete reference, and therefore reading through it is like reading through a dictionary -- not necessarily fun, but comprehensive
# 

# In[ ]:


# Write your own function definition


# In[ ]:


# Call your function several times to make sure it works


# ## Modules

# A module is like a library book containing code.  You can write your own module files that include functions you want to save or variables that you want to associate with the code.  
# 
# You can fetch a module using "import", and then all of its variables and functions will be usable inside the local code you are running.  It's therefore very useful for using code that others have already written.

# In[ ]:


import mymodule


# Below is the contents of mymodule

# In[ ]:


# %load 'mymodule.py'
def compound_calculator(p,r,y,c):
    '''
    compound_calculator(p,r,y,c) calculates the value at year y of an investment
    p = principal
    r = interest rate (percent value)
    y = year
    c = contribution at end of each year
    '''

    r = r/100
    return 'Final value is ${:,.2f}'.format(p*(1 + r)**y + c*( ((1 + r)**(y+1) - (1 + r)) / r ))

holyhandgrenade = ['one','two','five']

riddleanswers = {'name':'Lancelot', 'quest':'Holy Grail', 'favorite colour':'blue'}

stocksDict = {'AAPL': 'Apple Inc.',
 'AMGN': 'Amgen Inc.',
 'AXP': 'American Express Company',
 'BA': 'The Boeing Company',
 'CAT': 'Caterpillar Inc.',
 'CRM': 'Salesforce',
 'CSCO': 'Cisco Systems, Inc.',
 'CVX': 'Chevron Corporation',
 'DIS': 'The Walt Disney Company',
 'DOW': 'Dow Inc.',
 'GS': 'The Goldman Sachs Group, Inc.',
 'HD': 'The Home Depot, Inc.',
 'HON': 'Honeywell International Inc.',
 'IBM': 'International Business Machines Corporation',
 'INTC': 'Intel Corporation',
 'JNJ': 'Johnson & Johnson',
 'JPM': 'JPMorgan Chase & Co.',
 'KO': 'The Coca-Cola Company',
 'MCD': "McDonald's Corporation",
 'MMM': '3M Company',
 'MRK': 'Merck & Co., Inc.',
 'MSFT': 'Microsoft Corporation',
 'NKE': 'NIKE, Inc.',
 'PG': 'The Procter & Gamble Company',
 'TRV': 'The Travelers Companies, Inc.',
 'UNH': 'UnitedHealth Group Incorporated',
 'V': 'Visa Inc.',
 'VZ': 'Verizon Communications Inc.',
 'WBA': 'Walgreens Boots Alliance, Inc.',
 'WMT': 'Walmart Inc.'}



# In[ ]:


dir(mymodule)


# In[ ]:


mymodule.holyhandgrenade


# In[ ]:


for i in range(len(mymodule.holyhandgrenade)):
    print(mymodule.holyhandgrenade[i])


# Importing certain functions from the module with an abbreviated name

# In[ ]:


from mymodule import holyhandgrenade as hhg


# In[ ]:


hhg


# In[ ]:


import mymodule as mm


# In[ ]:


mm.hhg


# In[ ]:


mm.holyhandgrenade


# In[ ]:


help(mymodule.compound_calculator)


# In[ ]:


print(mymodule.compound_calculator.__doc__)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'mymodule.compound_calculator')


# In[ ]:


mymodule.compound_calculator(1000,5,1,100)


# In[ ]:


mm.riddleanswers


# In[ ]:


mm.stocksDict


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


t = np.linspace(0, 2*np.pi, 1000)


# In[ ]:


t


# In[ ]:


y = np.sin(t)


# In[ ]:


y


# In[ ]:


plt.plot(t,y);


# In[ ]:


fig,ax = plt.subplots()
plt.plot(t,y)
plt.xlim(0,2*np.pi)
plt.xlabel('t',fontsize=14)
plt.ylabel('y',fontsize=14)
ax.text(7,0, 'We are plotting the equation $y(t) = \sin(t)$', fontsize=14);

