#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
# import the regressor 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import tree
from scipy.stats import norm, skew 
from scipy.special import boxcox1p
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


# In[2]:


os.chdir('E:')
#file = pd.read_excel('E:\\eval.xlsx')
file = pd.read_excel('E:\\file.xlsx')


# In[3]:


file.shape


# In[4]:


file.info()


# In[5]:


file.describe()


# In[6]:


file.head(10)


# In[7]:


#handling missing data
all_data_na = (file.isnull().sum() / len(file)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()


# In[8]:


file["calcu mmr"] = file.groupby("State")["calcu mmr"].transform(
    lambda x: x.fillna(x.mean()))
file["phc per 100000"] = file.groupby("State")["phc per 100000"].transform(
    lambda x: x.fillna(x.mean()))
file["per home skill"] = file.groupby("State")["per home skill"].transform(
    lambda x: x.fillna(x.mean()))


# In[9]:


#handling missing data
all_data_na = (file.isnull().sum() / len(file)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()


# In[10]:


cols = ['calcu mmr', 'calcu IMR 12', 'literacy', 'phc per 100000', 'percent insti', 'per home skill', 'per capita health exp', 'anc', 'anc check', 'pnc per live birth', 'haemo']
data = file.loc[:, cols]
data.info()


# In[11]:


data.head()


# In[12]:


numeric_feats = data.dtypes[data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


# In[13]:


#box-cox transformation 
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0
for feat in skewed_features:
    #data[feat] += 1
    data[feat] = boxcox1p(data[feat], lam)
    
#data[skewed_features] = np.log1p(data[skewed_features])


# In[14]:


data.describe()


# In[15]:


corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)


# In[16]:


sns.set()
sns.pairplot(data[cols], size=2.5)
plt.show()


# In[17]:


fig, ax = plt.subplots()
ax.scatter(data['calcu mmr'], data['calcu IMR 12'])
plt.ylabel('Infant Mortality Ratio', fontsize=13)
plt.xlabel('Maternal Mortality ratio', fontsize=13)
plt.show()


# In[34]:


fig, ax = plt.subplots()
ax.scatter(data['percent insti'], data['calcu mmr'])
plt.ylabel('Maternal Mortality Rate', fontsize=13)
plt.xlabel('Percent Institutional Deliveries', fontsize=13)
plt.show()


# In[35]:


fig, ax = plt.subplots()
ax.scatter(data['per capita health exp'], data['calcu mmr'])
plt.ylabel('Maternal Mortality Rate', fontsize=13)
plt.xlabel('Per Capita Health Expenditure', fontsize=13)
plt.show()


# In[18]:


fig, ax = plt.subplots()
ax.scatter(data['literacy'], data['calcu mmr'])
plt.ylabel('Maternal Mortality Rate', fontsize=13)
plt.xlabel('literacy rate', fontsize=13)
plt.show()


# In[19]:


feature_cols = ['literacy', 'percent insti', 'per home skill', 'pnc per live birth', 'per capita health exp']
x = data[feature_cols]
y = data['calcu mmr']
  
# create a regressor object 
reg = DecisionTreeRegressor(random_state = 0, max_depth = 3)  
  
# fit the regressor with X and Y data 
model = reg.fit(x, y) 

fig = plt.figure(figsize=(50,10))
tree.plot_tree(reg, feature_names = feature_cols,  class_names = 'calcu mmr', filled=True)


# In[20]:


feature_cols = ['percent insti', 'per home skill']
x = data[feature_cols]
y = data['calcu mmr']
  
# create a regressor object 
reg = DecisionTreeRegressor(random_state = 0, max_depth = 3)  
  
# fit the regressor with X and Y data 
model = reg.fit(x, y) 

fig = plt.figure(figsize=(50,10))
tree.plot_tree(reg, feature_names = feature_cols,  class_names = 'calcu mmr', filled=True)


# In[21]:


feature_cols = ['pnc per live birth', 'per capita health exp']
x = data[feature_cols]
y = data['calcu mmr']
  
# create a regressor object 
reg = DecisionTreeRegressor(random_state = 0, max_depth = 3)  
  
# fit the regressor with X and Y data 
model = reg.fit(x, y) 

fig = plt.figure(figsize=(50,10))
tree.plot_tree(reg, feature_names = feature_cols,  class_names = 'calcu mmr', filled=True)


# In[24]:


fig, ax = plt.subplots()
ax.scatter(data['haemo'], data['calcu mmr'])
plt.ylabel('Maternal Mortality Ratio', fontsize=13)
plt.xlabel('Percent women with haemoglobin less than 7', fontsize=13)
plt.show()


# In[25]:



haemo =  data['haemo'].values.reshape(-1, 1)
mmr = data['calcu mmr'].values.reshape(-1, 1)
regressor = LinearRegression()  
regressor.fit(haemo, mmr)


# In[26]:


#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)


# In[27]:


feature_cols = ['literacy', 'per capita health exp', 'phc per 100000']
x = data[feature_cols]
y = data['haemo']
  
# create a regressor object 
reg = DecisionTreeRegressor(random_state = 0, max_depth = 3)  
  
# fit the regressor with X and Y data 
model = reg.fit(x, y) 

fig = plt.figure(figsize=(50,10))
tree.plot_tree(reg, feature_names = feature_cols,  class_names = 'haemo', filled=True)


# In[28]:


feature_cols = ['calcu mmr', 'pnc per live birth']
x = data[feature_cols]
y = data['calcu IMR 12']
  
# create a regressor object 
reg = DecisionTreeRegressor(random_state = 0, max_depth = 3)  
  
# fit the regressor with X and Y data 
model = reg.fit(x, y) 

fig = plt.figure(figsize=(50,10))
tree.plot_tree(reg, feature_names = feature_cols,  class_names = 'calcu IMR 12', filled=True)


# In[29]:


imr =  data['calcu IMR 12'].values.reshape(-1, 1)
mmr = data['calcu mmr'].values.reshape(-1, 1)
regressor = LinearRegression()  
regressor.fit(mmr, imr)


# In[30]:


#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)


# In[32]:


#x = daata["RM"] ## X usually means our input variables (or independent variables)
#y = target["MEDV"] ## Y usually means our output/dependent variable
imr =  data['calcu IMR 12'].values.reshape(-1, 1)
mmr = data['calcu mmr'].values.reshape(-1, 1)
mmr = sm.add_constant(mmr) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(imr, mmr).fit() ## sm.OLS(output, input)

# Print out the statistics
model.summary()


# In[36]:


#x = daata["RM"] ## X usually means our input variables (or independent variables)
#y = target["MEDV"] ## Y usually means our output/dependent variable
yy =  data['calcu mmr'].values.reshape(-1, 1)
xx = data['haemo'].values.reshape(-1, 1)
xx = sm.add_constant(xx) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(yy, xx).fit() ## sm.OLS(output, input)

# Print out the statistics
model.summary()

