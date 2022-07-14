#!/usr/bin/env python
# coding: utf-8

# ## Example:  Grabbing Time Series Data from a Public API (here Yahoo finance)
# 
# "Yahoo_fin is a Python 3 package designed to scrape historical stock price data, as well as to provide current information on market caps, dividend yields, and which stocks comprise the major exchanges. Additional functionality includes scraping income statements, balance sheets, cash flows, holder information, and analyst data. The package includes the ability to scrape live (real-time) stock prices, capture cryptocurrency data, and get the most actively traded stocks on a current trading day. Yahoo_fin also contains a module for retrieving option prices and expiration dates." 
# 
# -- [yahoo_fin documentation](http://theautomatic.net/yahoo_fin-documentation/)

# In[1]:


import yahoo_fin.stock_info as si
import requests
import matplotlib.pyplot as plt
import ipywidgets


# [First one must know how to access and use the API... I'll ignore that]

# In[ ]:


# Getting the actual company name from a ticker symbol
def get_symbol(symbol):
    url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)
    result = requests.get(url).json()
    for x in result['ResultSet']['Result']:
        if x['symbol'] == symbol:
            return x['name']


# In[ ]:


print(get_symbol('AAPL'))


# In[ ]:


few_days = si.get_data('aapl', start_date = '01/01/2020', end_date = '11/30/2020')


# In[ ]:


few_days


# In[ ]:


fig,ax = plt.subplots(1,1,figsize=(7,5))
ax.plot(few_days.index, few_days.high)
ax.set_title(get_symbol('AAPL'))
fig.autofmt_xdate()


# In[ ]:


dow_list = si.tickers_dow()


# In[ ]:


def plotdows(ticker='AAPL'):
    few_days = si.get_data(ticker, start_date = '01/01/2020', end_date = '11/30/2020')
    fig,ax = plt.subplots(1,1,figsize=(7,5))
    ax.plot(few_days.index, few_days.high)
    ax.set_title(get_symbol(ticker))
    fig.autofmt_xdate()
    
ipywidgets.interact(plotdows, ticker=dow_list);

