#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# StandardScaler function is used to standardize data in a comman range to easily understand it.
from sklearn.preprocessing import StandardScaler
from sklearn import svm # Support vector machine
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score # give accuracy


# In[3]:


# loading data from csv file to a pandas dataframe
parkinson_data = pd.read_csv(r"C:\Users\dell\Desktop\8TH SEM\Capstone_project\DATASETS\parkinsons.csv")


# In[4]:


# printing first 5 rows of dataframe
parkinson_data.head()


# In[5]:


# number of rows and columns in dataframe
parkinson_data.shape


# In[6]:


# getting more info about data
parkinson_data.info()


# In[7]:


# checking for missing values in each column
parkinson_data.isnull().sum()


# In[8]:


# getting some statistical measures about data
parkinson_data.describe()


# In[9]:


# distribution of target_column
parkinson_data['status'].value_counts()


# In[10]:


# grouping data based on status
parkinson_data.groupby('status').mean()


# In[11]:


# making pie chart and showing the status report of parkinson and healthy person in pie graph form using pyplot library
parkinson_data['status'].value_counts().plot(kind='pie', autopct = "%1.0f%%")


# In[12]:


# Split dataset into features (X) and target (y)
# target_column is status because we have to check using the voice recording sample of the patients
#whether they are having parkinson's or not
X = parkinson_data.copy()
X = X.drop(columns = ['name','status'], axis=1)
y = parkinson_data['status']
print(X)
print(y)


# In[13]:


parkinson_data.boxplot(figsize=(15,7))


# In[14]:


# Split dataset into training and testing sets
# here, training data is 80% and testing data is 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# In[15]:


print(X.shape, X_train.shape, X_test.shape)


# In[16]:


# Data standardization : to make data in same range but it wouldn't change the meaning of data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(X_train)


# In[16]:


# SVC(Support Vector Classifier) classifies the data
model = svm.SVC(kernel = "linear", C=1, gamma='scale')
# training svm model with training data
model.fit(X_train, y_train)


# In[17]:


# Evaluating SVM on training data to find accuracy_score
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(y_train, X_train_prediction)
print("Accuracy score of training data : ", training_data_accuracy*100)


# Evaluating SVM on testing data to find accuracy_score
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(y_test, X_test_prediction)
print("Accuracy score of testing data : ", testing_data_accuracy*100)


# In[18]:


# Initialize a Random Forest classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=2)

# Fit the model on the training data
rfc.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rfc.predict(X_test)
# Calculate the accuracy of the model
testing_accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of testing data : ",testing_accuracy*100)


X_train_prediction = rfc.predict(X_train)
training_data_accuracy = accuracy_score(y_train, X_train_prediction)
print("Accuracy score of training data : ", training_data_accuracy*100)


# In[19]:


# Create a KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier on the training data
knn.fit(X_train, y_train)
# Make predictions on the test data
y_pred = knn.predict(X_test)

# Evaluate the accuracy of the classifier
print("Accuracy of testing data of KNN algorithm:",(accuracy_score(y_test, y_pred))*100)

X_train_prediction = knn.predict(X_train)
training_data_accuracy = accuracy_score(y_train, X_train_prediction)
print("Accuracy score of training data : ", training_data_accuracy*100)


# In[20]:


parkinson_data.hist()

