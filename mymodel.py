#!/usr/bin/env python
# coding: utf-8

# In[22]:


import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score,f1_score,precision_score

from sklearn import metrics

df = pd.read_csv ('heart.csv')


# In[17]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
X = df.drop(['target'], axis=1)
y = df.target
X_std = StandardScaler().fit_transform(X)

# In[18]:


x_train, x_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=433)


# In[19]:


model = LogisticRegression(C=0.1)
model.fit(x_train,y_train)
accuracy = model.score(x_test,y_test)
print('Logistic Regression Accuracy -->',((accuracy)*100))


# In[24]:


import mlflow
import mlflow.sklearn
from urllib.parse import parse_qsl, urljoin, urlparse
mlflow.set_tracking_uri("sqlite:///mymodel.db")


# In[25]:


with mlflow.start_run():
    
    model = LogisticRegression(C=0.1)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    accuracy = model.score(x_test,y_test)
    recall=recall_score(y_test, y_pred)
    f1=f1_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
 

    mlflow.log_param("accuracy", accuracy)
    mlflow.log_param("recall", recall)
    mlflow.log_param("f1 score", f1)
    mlflow.log_param("precision", precision)


    mlflow.sklearn.log_model(model, "model")
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
  #  model = mlflow.pyfunc.load_model(model_path)

    #if tracking_url_type_store != "file":
     #   mlflow.sklearn.log_model(model, "model", registered_model_name="Logistic Regression Model")
    #else:
      #  mlflow.sklearn.log_model(model, "model")


# In[ ]:




