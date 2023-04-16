#!/usr/bin/env python
# coding: utf-8

# In[71]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Uploading the data

# In[72]:


train_data=pd.read_csv(r"C:\Users\meena\Downloads\illegal_fishing_trn_data.csv")
test_data=pd.read_csv(r"C:\Users\meena\Downloads\illegal_fishing_tst_data.csv")
traindata_classlabel=pd.read_csv(r"C:\Users\meena\Downloads\trn_classlabel.csv")


# In[4]:


train_data


# In[5]:


traindata_classlabel


# In[6]:


train_data.info()


# In[7]:


traindata_classlabel.info()


# ### Counting the target variables

# In[8]:


col="class"
count= traindata_classlabel.groupby('class').size()
print("Frequency of values in column", col, "is:", count)


# ## EDA & Classification including -1 Class 

# In[9]:


train_data.describe()


# In[10]:


traindata_classlabel.describe()


# ### Adding the classlabel to train data

# In[11]:


train_data['class']=traindata_classlabel


# In[12]:


train_data


# ### Checking & Handling missing values

# In[13]:


train_data.isnull().sum()


# ### Droping the null values rows

# In[14]:


train_data.dropna(inplace=True)


# ## Exploratory data analysis

# In[15]:


train_data.corr()


# In[21]:


fig = plt.figure(figsize=(10,8))
sns.heatmap(train_data.corr())


# In[26]:


plt.figure(figsize=(20,8))
sns.barplot(data=train_data,ci='sd')


# In[29]:


sns.pairplot(data=train_data)


# ### Droping the classlabel

# In[16]:


X=train_data.drop(['class'],axis=1)
y=train_data['class']


# In[17]:


X.shape, y.shape


# ### Checking partition of class labels in percentage

# In[20]:


y.value_counts().plot.pie(autopct='%.2f')


# As we can see most of the target variable are -1 and its dominating the other target variables and causing overfitting , So
# we need handle the imbalance data.

# ## Classifiaction without hyperparamuter tuning and without sampliing the data

# #### Splitting the dataset into the training and test set

# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=20, stratify=y)


# In[87]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# ### Creating & training KNeighborsClassifier

# In[122]:


from sklearn.neighbors import KNeighborsClassifier

m1 = KNeighborsClassifier()
m1.fit(X_train,y_train)
y_train_pred = m1.predict(X_train)
y_test_pred = m1.predict(X_test)

print("Train Set Accuracy:"+str(accuracy_score(y_train_pred,y_train)*100))
print("Test Set Accuracy:"+str(accuracy_score(y_test_pred,y_test)*100))
print("\nConfusion Matrix:\n%s"%confusion_matrix(y_test_pred,y_test))
print("\nClassification Report:\n%s"%classification_report(y_test_pred,y_test))


# ### Creating & training DecisionTreeClassifier

# In[80]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report

m1 = DecisionTreeClassifier()
m1.fit(X_train,y_train)
y_train_pred = m1.predict(X_train)
y_test_pred = m1.predict(X_test)

print("Train Set Accuracy:"+str(accuracy_score(y_train_pred,y_train)*100))
print("Test Set Accuracy:"+str(accuracy_score(y_test_pred,y_test)*100))
print("\nConfusion Matrix:\n%s"%confusion_matrix(y_test_pred,y_test))
print("\nClassification Report:\n%s"%classification_report(y_test_pred,y_test))


# ### Creating & training RandomForestClassifier

# In[29]:


from sklearn.ensemble import RandomForestClassifier
m3 = RandomForestClassifier()
m3.fit(X_train,y_train)
y_train_pred = m3.predict(X_train)
y_test_pred = m3.predict(X_test)

print("Train Set Accuracy:"+str(accuracy_score(y_train_pred,y_train)*100))
print("Test Set Accuracy:"+str(accuracy_score(y_test_pred,y_test)*100))
print("\nConfusion Matrix:\n%s"%confusion_matrix(y_test_pred,y_test))
print("\nClassification Report:\n%s"%classification_report(y_test_pred,y_test))


# ## Classifiaction with hyperparamuter tuning without sampliing the data

# ### GridSearchCV with Random Forest Classifier

# In[29]:


#Importing gridsearchcv & setting the parameters


# In[31]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import GridSearchCV

n_estimators = [5,20,50,100]
max_depth=[3,5,7,None]
max_samples=[.2,.5,.75,.1]

param_grid = {
    'n_estimators' : [5,20,50,100],
'max_depth':[3,5,7,None],
'max_samples':[.2,.5,.75,.1]
}


rfc = RandomForestClassifier()

rfc_grid = GridSearchCV(estimator = rfc,
                        param_grid = param_grid,
                        cv=5,
                        n_jobs= -1,
                        
)

rfc_grid.fit(X_train,y_train)

print("Best hyperparameters:", rfc_grid.best_params_)

prediction_rfc_grid = rfc_grid.predict(X_test)
print("Accuracy Score:\n",accuracy_score(y_test, prediction_rfc_grid))
print("Confusion Matrix:\n", confusion_matrix(y_test,prediction_rfc_grid))
print("Classification Report:\n", classification_report(y_test,prediction_rfc_grid))


# ### GridSearchCV with Decision Forest Classifier

# In[ ]:


#Importing gridsearchcv & setting the parameters


# In[23]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import GridSearchCV
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [1,2,3,4,5,6,7,8,9,10,None]
    
}

dt = DecisionTreeClassifier()

dt_grid = GridSearchCV(dt, param_grid, cv=5, n_jobs = -1)
dt_grid.fit(X_train, y_train)
print("Best Hyperparameters:", dt_grid.best_params_)
prediction_dt_grid=dt_grid.predict(X_test)
print("Accuracy Score:\n", accuracy_score(y_test,prediction_dt_grid))
print("Confusion Matrix:\n", confusion_matrix(y_test,prediction_dt_grid))
print("Classification Report:\n", classification_report(y_test,prediction_dt_grid))



# ### GridSearchCV with KNeighbors Classifier (KNN)

# In[30]:


#Importing classifier & gridsearchcv & setting the parameters


# In[32]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': [3, 5, 7, 9, 11],
              'weights': ['uniform', 'distance']}

knn = KNeighborsClassifier()

grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs= -1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_score = grid_search.best_score_

final= knn.set_params(**best_params)
final.fit(X_train,y_train)
y_pred=final.predict(X_test)

print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(best_params)


# ## Importing the test data

# In[126]:


test_data=pd.read_csv(r"C:\Users\meena\Downloads\illegal_fishing_tst_data.csv")
test_data.shape


# In[127]:


#Handling the missing values


# In[128]:


test_data.isnull().sum()


# In[129]:


#Droping the missing values


# In[130]:


test_data.dropna(inplace=True)


# In[131]:


class_=knn.predict(test_data)


# In[132]:


class_


# In[134]:


sns.countplot(x=class_)

