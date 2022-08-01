#!/usr/bin/env python
# coding: utf-8

# # Python Basics

# Python is a powerful language with many useful features. 

# ## Hello World!

# In[1]:


print("Hello world!")


# This command used the `print()` function with the string `Hello world!` as the value for its input parameter.

# ## Variables and literals

# A variable is a named location in memory.  For example:

# In[2]:


a = 42


# `a` is the variable, and its value is now set to 42.

# In[3]:


print(a)


# Note that variables do not need to be defined with a specific type in Python.
# 
# Also, in Jupyter notebooks, the value of a variable will be output if the variable sits alone on the last line of a cell.

# In[4]:


a


# In[5]:


a = "forty two"


# In[6]:


print('The print function can take two or more input parameters: a =', a)


# ## Operators

# Python has standard arithmetic operators for common math operations:

# In[7]:


x = 42
y = 24

# Add two operands
print('x + y =', x+y) # Output: x + y = 66

# Subtract right operand from the left
print('x - y =', x-y) # Output: x - y = 18

# Multiply two operands
print('x * y =', x*y) # Output: x * y = 1008

# Divide left operand by the right one 
print('x / y =', x/y) # Output: x / y = 1.75

# Floor division (quotient)
print('x // y =', x//y) # Output: x // y = 1

# Remainder of the division of left operand by the right
print('x % y =', x%y) # Output: x % y = 18

# Left operand raised to the power of right (x^y)
print('x ** y =', x**y) # Output: x ** y = 907784931546351634835748413459499319296


# # Comments

# As you see above, Python has special notation for comments:

# In[8]:


# This is a single line comment


# In[9]:


# A two line
# comment


# In[10]:


'''
A multi-line comment
for printing 
print('Hi')
'''
print('Bye')


# In[11]:


"""
A multi-line comment
for printing 
print('Hi')
"""
print('Bye')


# ## Data structures

# Two of the most common data structures in Python are lists and dictionaries.

# ### Lists

# In[12]:


# empty list
my_list = []

# list of integers
my_list = [1, 2, 5]

# list with mixed data types
my_list = [1, "Parrot", 3.14159]


# * List
#     * an ordered collection of elements
#     * elements are placed in square brackets `[]` and separated by commas
#     * they can have elements of different types
#     * the index operator `[]` with an integer index will retrieve an element at that index
#     * negative indices can be used to count from the end of the list

# In[13]:


riddles = ["Lancelot", "Holy Grail", "Blue", "Southwest"]

# Accessing first element
print(riddles[0])

# Accessing fourth element
print(riddles[3])


# ### Dictionaries

# In[14]:


# empty dictionary
my_dict = {}

# dictionary with integer keys
my_dict = {1: 'red', 2: 'blue'}

# dictionary with mixed keys
my_dict = {'name': 'Arthur', 1: [1, 2, 5]}


# * Dictionary
#     * an unordered collection of elements
#     * elements consist of `key: value` pairs
#     * elements are placed in curly braces `{}` and separated by commas
#     * they can have elements of different types
#     * the index operator `[]` with a key index will retrieve the value corresponding to that key    

# In[15]:


person = {'name':'Gwen', 'age': 26, 'salary': 94534.2}
print(person['age']) # Output: 26


# In[16]:


person = {'name':'Gwen', 'age': 26}

# Changing age to 36
person['age'] = 36 
print(person) # Output: {'name': 'Gwen', 'age': 36}

# Adding salary key, value pair
person['salary'] = 94342.4
print(person) # Output: {'name': 'Gwen', 'age': 36, 'salary': 94342.4}

# Deleting age
del person['age']
print(person) # Output: {'name': 'Gwen', 'salary': 94342.4}

# Deleting entire dictionary
del person


# ### Strings
# * Strings
#     * an ordered collection of characters
#     * immutable
#     * like lists, you can use the index operator `[]` with an integer index to retrieve an element at that index

# In[17]:


# all of the following are equivalent
my_string = 'Arthur'
print(my_string)

my_string = "Arthur"
print(my_string)

my_string = '''Arthur'''
print(my_string)

# triple quotes string can extend multiple lines
my_string = """Hello, Arthur, welcome to
           the world of [Monty] Python"""
print(my_string)


# In[18]:


str0 = 'e.t. phone home!'
print('str0 = ', str0) # Output: e.t. phone home!

print('str0[0] = ', str0[0]) # Output: e

print('str0[-1] = ', str0[-1]) # Output: !

#slicing 2nd to 5th character
print('str0[1:5] = ', str0[1:5]) # Output: .t. 

#slicing 6th to 2nd last character
print('str0[5:-2] = ', str0[5:-2]) # Output: phone hom


# You can use the operators `+` and `*` to concatenate and duplicate strings:

# In[19]:


str1 = 'Phone '
str2 ='Home!'

# Output: Phone Home!
print(str1 + str2)

# Phone Phone Phone
print(str1 * 3)


# ## Flow control

# ### If/else statements
# `if`, `elif`, and `else` allow you to execute Python commands only if some condition is satisfied.

# In[20]:


num = -1

if num > 0:
    print("Positive number")
elif num == 0:
    print("Zero")
else:
    print("Negative number")
    
# Output: Negative number


# **Note:** Python uses indentation to define blocks of code
# * this is not merely a matter of style in Python
# * a code block starts with indentation and ends with the first unindented line
# * it is up to you how many spaces you want to use for indentation, as long as you are consistent
# * you can not mix tabs and spaces -- here in Jupyter, the indentation is set automatically to 4 spaces
#     * generally, four whitespaces are used for indentation and is preferred over tabs.

# In[21]:


if False:
  print("I am inside the body of if.")
  print("I am also inside the body of if.")
print("I am outside the body of if")

# Output: I am outside the body of if.


# Mathematical conditions:
# * These are rather straight-forward, but try making a few if/elif/else statements to check the True and False values.
# * Equals: `==`
# * Does not equal: `!=`
# * Less than: `<`
# * Less than or equal to: `<=`
# * Greater than: `>`
# * Greater than or equal to: `>=`
# 

# Boolean combinations:  You can evaluate more than one True/False condition in the statements by combining conditions with `and`, `or`, and `not`

# In[22]:


# Switch the value of claim to check the logic

claim = 5
if claim == 1 or claim == 2:
    print('do not throw the grenade yet')
elif claim == 3:
    print('throw the grenade')
elif claim == 5:
    print('Silly Arthur.  3 comes after 2.')
else:
    print('You are too late - kaboom!')


# ### for loops

# In Python, a `for` loop is used to iterate over a sequence (list, string, etc) or other iterable objects. Iterating over a sequence is called traversal.
# 
# Here's an example to find the sum of all numbers stored in a list.

# In[23]:


numbers = [1, 1, 2, 3, 5, 8]

sum = 0

# iterate over the list
for val in numbers:
  sum = sum+val

print("The sum is", sum) # Output: The sum is 20


# `for` loops can also be used to iterate a given number of times with the special function `range`:
# * `range(n)` returns a sequence of numbers from 0 to n (n not included)
# * `range(n,m)` returns a sequence of numbers from n to m (m not included)
# * `range(n,m,i)` returns a sequence of numbers from n to m (m not included) with an interval of i between numbers

# In[24]:


for disneytrip in range(10):
    print(str(disneytrip) + ': Are we there yet?')


# ## Functions

# A function is a group of related statements that perform a specific task. 
# 
# * Functions start with the `def` keyword,
# * then the function name, 
# * followed immediately by parentheses that enclose a parameter list,
# * and then a `:`
# * The function body falls on subsequent lines
# * The function ends when the indentation stops

# In[25]:


def f():
    print('Hello World!')


# In[26]:


# You need to call it to see any output or retrieve returned values
f()


# In[27]:


def f2(a):
    return a*2


# In[28]:


# This will return an error
f2()


# In[ ]:


f2(4)


# Let's consider a mathematical operation that we wouldn't want to keep typing every time when run it.
# 
# The following is an equation for calculating compound interest with annual contributions
# 
# * p = principal
# * r = annual interest rate in percent
# * y = year of the balance
# * c = annual contribution (made at the start of the year)
# 
# $$\text{Balance}(y) = p(1 + r)^y + c\left[\frac{(1 + r)^{y+1} - (1 + r)}{r} \right]$$

# In[ ]:


def f(p,r,y,c):
    return p*(1 + r)**y + c*( ((1 + r)**(y+1) - (1 + r)) / r )


# In[ ]:


year = 1
principal = 1000
rate = 5
annual_contribution = 100
f(principal, rate, year, annual_contribution)


# Functions are useful, of course, for more complicated machinations.

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


# ## Methods

# Variables in Python are actually objects (yes, in the object-oriented sense of object).
# 
# Python variables can therefore have attributes and methods associated with them.

# In[ ]:


numbers


# In[ ]:


type(numbers)


# In[ ]:


numbers.index(5)


# The `.` notation is used to denote that we are using the `index` method associated with the `numbers` list variable, and passing `5` as the input parameter to that method.
# 
# It tells us that the value `5` is in the `numbers` list at index 4.

# In[ ]:


numbers[4]


# The `reverse` method will reverse the elements in-place.

# In[ ]:


numbers.reverse()


# In[ ]:


numbers


# Strings and dictionaries also have methods.

# In[ ]:


# 'split' splits a string into a list of elements
# the splitting happens by default on whitespace

'My name is Ben'.split()


# In[ ]:


# 'join' is a string method that creates a single string from a list

'-'.join(['Combo','words','are','in','this','list'])


# ## Modules

# A lot of coders have written Python code that you can easily reuse.  

# In[ ]:


# Example:

# retrieve the `math` module
import math

# use constants stored in the module
print('pi = ', math.pi)

# use functions written in the module
print('The value of sin(pi/2) is', math.sin(math.pi/2))


# Some code comes standard with every Python installation.  Other code needs to be retrieved and installed.  However, once you have the code, it can dramatically expand your coding capabilities.
# 
# Modules allow us to use externally developed code.
# * A module is a group of code items such as functions that are related to one another. Individual modules are often grouped together as a library.
# * Modules can be loaded using `import <modulename>`. 
# * Functions that are part of the module `modulename` can then be used by typing `modulename.functionname()`. 
#   * For example, `sin()` is a function that is part of the `math` module
#   * We can use to by typing `math.sin()` with some number between the parentheses.
# * Modules may also contain constants in addition to functions.
#   * The `math` module includes a constant for $\pi$ -- `math.pi`

# We can even write our own modules!
# 
# Open the included "mymodule.py" file and look it over before executing the following cells.

# In[ ]:


import mymodule


# In[ ]:


mymodule.holyhandgrenade


# In[ ]:


mymodule.compound_calculator(1000,5,1,100)


# One can import select functions or variables from a module, as well as rename them with an alias.

# In[ ]:


import mymodule as mm


# In[ ]:


mm.stocksDict

