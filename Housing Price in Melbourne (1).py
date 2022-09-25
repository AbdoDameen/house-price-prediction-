#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


Housing_Melb = pd.read_excel('/home/Housing prices.xlsx')


# In[3]:


Housing_Melb.head(5)


# In[4]:


Housing_Melb.info()


# In[5]:


Housing_Melb.describe()


# In[6]:


Housing_Melb['Location'].value_counts()


# In[7]:





# In[8]:


list(Housing_Melb.columns)


# In[9]:


plt.figure(figsize=(15,8))
ax = sns.scatterplot(x="Number of Rooms", y="Selling Price", data=Housing_Melb)
ax.set_title("Selling Price vs. Number of Rooms");


# In[10]:



fig, ax = plt.subplots(figsize=(15,8))
scat = ax.scatter(y=Housing_Melb['Selling Price'],x=Housing_Melb["Number of Rooms"], c=Housing_Melb["Location"], marker='o')
fig.colorbar(scat)


plt.show()


# **add a best-fit line to a scatterplot**

# In[11]:


#plt.figure(figsize=(15,8))
ax = sns.lmplot(x="Number of Rooms", y="Selling Price", data=Housing_Melb)

ax.fig.set_figwidth(14.27)
ax.fig.set_figheight(8.7)


# **Adding color as a third dimension** 
# 

# In[12]:


ax = sns.lmplot(x="Number of Rooms", y="Selling Price", hue="Location", data=Housing_Melb)
ax.fig.set_figwidth(14.27)
ax.fig.set_figheight(8.7);


# In[13]:


from scipy import stats
stats.pearsonr(Housing_Melb['Selling Price'], Housing_Melb['Number of Rooms'])


# In[14]:


plt.figure(figsize=(15,9))
ax = sns.boxplot(x="Number of Rooms", y='Selling Price', data=Housing_Melb, color='#99c2a2')
ax = sns.swarmplot(x="Number of Rooms", y='Selling Price', data=Housing_Melb, color='#7d0013')
plt.show()


# In[15]:


plt.figure(figsize=(15,9))
sns.barplot(data=Housing_Melb, x="Number of Rooms", y='Selling Price')
plt.show()


# var = 'OverallQual'
# data = Housing_Melb
# f, ax = plt.subplots(figsize=(14, 8))
# fig = sns.boxplot(x=var, y="Selling Price", data=data)
# fig.axis(ymin=0, ymax=800000);

# 

# In[16]:


plt.figure(figsize=(15,9))
sns.barplot(data=Housing_Melb, x="Number of Rooms", y='Selling Price', hue="Location")
plt.show()


# In[17]:


from sklearn import linear_model
x = Housing_Melb[["Number of Rooms","Location"]]
y = Housing_Melb['Selling Price']
regr = linear_model.LinearRegression()
regr.fit(x, y)
regr.score(x, y)
print(f'The intercept is:', regr.intercept_)
print(f'The slopes are:', regr.coef_)
print(f'The score is', regr.score(x, y))


# In[18]:


import statsmodels.api as sm
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
print(model.summary())


# In[19]:


predicted = regr.predict([[9, 0]])

print(predicted)


# In[20]:


from scipy import stats
stats.pearsonr(Housing_Melb['Selling Price'], Housing_Melb['Number of Rooms'])


# In[21]:


cormat = Housing_Melb.corr()
round(cormat,2)


# In[22]:


fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
sns.heatmap(cormat, annot=True, linewidths=.5, ax=ax)

#sns.heatmap(cormat);


# In[23]:





# In[24]:


import scipy.stats as stats
# stats f_oneway functions takes the groups as input and returns ANOVA F and p value
fvalue, pvalue = stats.f_oneway(Housing_Melb['Selling Price'],Housing_Melb["Number of Rooms"], Housing_Melb["Location"])
print(fvalue, pvalue)


# In[25]:


sp = Housing_Melb['Selling Price']
nr = Housing_Melb["Number of Rooms"]
lo = Housing_Melb["Location"]
import scipy.stats as st

def f_test(sp, nr, alt="two_sided"):
   
    df1 = len(sp) - 1
    df2 = len(nr) - 1
    f = sp.var() / nr.var()
    if alt == "greater":
        p = 1.0 - st.f.cdf(f, df1, df2)
    elif alt == "less":
        p = st.f.cdf(f, df1, df2)
    else:
        # two-sided by default
        # Crawley, the R book, p.355
        p = 2.0*(1.0 - st.f.cdf(f, df1, df2))
    return f, p


# In[26]:


f_test(sp, nr, 'two_sided')


# In[27]:


sp = Housing_Melb['Selling Price']
nr = Housing_Melb["Number of Rooms"]
lo = Housing_Melb["Location"]
import scipy.stats as st

def f_test(sp, lo, alt="two_sided"):
   
    df1 = len(sp) - 1
    df2 = len(lo) - 1
    f = sp.var() / lo.var()
    if alt == "greater":
        p = 1.0 - st.f.cdf(f, df1, df2)
    elif alt == "less":
        p = st.f.cdf(f, df1, df2)
    else:
        # two-sided by default
        # Crawley, the R book, p.355
        p = 2.0*(1.0 - st.f.cdf(f, df1, df2))
    return f, p


# In[28]:


f_test(sp, lo, 'two_sided')


# In[29]:


Housing_Melb['Inter_erm'] = Housing_Melb["Number of Rooms"] * Housing_Melb["Location"]
Housing_Melb


# In[30]:


x = Housing_Melb[["Number of Rooms","Location",'Inter_erm']]
y = Housing_Melb['Selling Price']
regr = linear_model.LinearRegression()
regr.fit(x, y)
regr.score(x, y)
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
print(model.summary())


# In[31]:





# In[32]:




