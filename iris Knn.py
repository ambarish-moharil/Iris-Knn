
# coding: utf-8

# In[2]:


from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# In[13]:


iris = datasets.load_iris()


# In[14]:


type(iris)


# In[15]:


print(iris.keys())


# In[32]:


X= iris.data
y = iris.target


# In[33]:


df_iris= pd.DataFrame(X, columns=iris.feature_names)


# In[34]:


df_iris.head()


# In[24]:


df_iris.shape


# In[35]:


df_iris.describe()


# In[36]:


pd.scatter_matrix(df_iris, c=y, figsize = [8,8], s=150, marker='D')


# In[39]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# In[42]:


knn = KNeighborsClassifier(n_neighbors=8)


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state = 21 , stratify = y)


# In[44]:


knn.fit(X_train , y_train)


# In[46]:


knn.predict(X_test)


# In[47]:


knn.score(X_test, y_test)

