
# coding: utf-8

# In[47]:


# importing all data science libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[48]:


# loading data
df = pd.read_csv('/Users/hp/downloads/all/train.csv')
df.head()


# In[49]:


# importing dependednt and independent variables in the dataset
X_train = df.iloc[:,1:-1].values
Y_train = df.iloc[:,14:15].values


# In[50]:


# data visualization
# building the correlation matrix
sns.heatmap(df.corr())


# In[51]:


# Encoding categorical data
#from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#labelencoder = LabelEncoder()
#X[:,3] = labelencoder.fit_transform(X[:,3])
#onehotencoder = OnehotEncoder(categorical_features=[3])
#X = onehotencoder.fir_transform(X).toarray()
#print(X[0])


# In[52]:


#avoiding dummy variable trap
#X=X[:,1:]


# In[53]:


#split the data into train and test data
#from sklearn.model_selection import train_test_split
#x_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[54]:


# fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
model_fit = LinearRegression()
model_fit.fit(X_train,Y_train)




# In[55]:


#loading test data

df_test = pd.read_csv('/Users/hp/downloads/all/test.csv')
df_test.head()


# In[56]:


# importing dependednt and independent variables in the dataset
X_test = df_test.iloc[:,1:].values
#print(X_test)


# In[64]:


Y_pred = model_fit.predict(X_test)
X_id = df_test.iloc[:,0:1]
np_array=np.column_stack((X_id,Y_pred))
df_result = pd.DataFrame(np_array,columns=['ID','medv'])
df_result.to_csv('/Users/hp/downloads/all/result.csv',index=False)

