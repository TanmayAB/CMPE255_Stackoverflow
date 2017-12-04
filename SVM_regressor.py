
# coding: utf-8

# In[18]:

import re
import string
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVR
from collections import Counter
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from math import fabs
from sklearn.metrics import f1_score,accuracy_score, mean_squared_error, mean_squared_log_error


# In[2]:

stop = set(stopwords.words('english'))
raw_data = pd.read_csv('StackOverflow1million.csv')


# #### Preprocessing Question Titles

# In[3]:

#remove punctuations from a string and convert to lower case
raw_data['questions_title'] = raw_data['questions_title'].apply(lambda x:''.join([i.lower() for i in x 
                                                  if i not in string.punctuation]))

#remove stop words
raw_data['questions_title'] = raw_data['questions_title'].apply(lambda x:' '.join([i for i in x.split(' ') 
                                                  if i not in stop]))

#removing digits
raw_data['questions_title'] = raw_data['questions_title'].apply(lambda x:' '.join([i.replace(r'[0-9]+','') for i in x.split(' ') ]))


raw_data['questions_title'] = raw_data['questions_title'].apply(lambda x: re.sub(r'\c\b', 'clang', x))


raw_data['questions_title'] = raw_data['questions_title'].apply(lambda x:x.lstrip())

raw_data['questions_title'] = raw_data['questions_title'].apply(lambda x:','.join([i for i in x.split()]))


# #### Preprocessing Question Tags

# In[4]:

#remove pipe and seperate tags
raw_data['questions_tags'] = raw_data['questions_tags'].apply(lambda x:','.join([i.lower() for i in x.split('|')]))
raw_data['questions_tags'] = raw_data['questions_tags'].apply(lambda x: re.sub(r'\c\b', 'clang', x))


# #### Vectorize the data

# In[5]:

vectorizer = CountVectorizer()
dataset_que_tag = vectorizer.fit_transform(raw_data.questions_tags)
dataset_que_title = vectorizer.fit_transform(raw_data.questions_title)
train_y = raw_data['time'].tolist()


# In[6]:

train_x = hstack((dataset_que_title, dataset_que_tag))


# In[7]:

#print dataset_que_title
#print dataset_que_tag
#print dataset_x
print dataset_que_title.shape
print dataset_que_tag.shape
print train_x.shape
train_x = csr_matrix(train_x)


# #### Apply Dimensionality reduction

# In[8]:

#svd = TruncatedSVD(n_components=3000)
#train_x= svd.fit_transform(train_x)


# In[9]:

#Applying K fold validation
#first 1000 train set
X_train, X_test1, Y_train, Y_test1 = train_test_split(
    train_x, train_y, test_size=1000, random_state=42)
#second 1000
X_train, X_test2, Y_train, Y_test2 = train_test_split(
    X_train, Y_train, test_size=1000, random_state=42)
#Third 1000
X_train, X_test3, Y_train, Y_test3 = train_test_split(
    X_train, Y_train, test_size=1000, random_state=42)
#forth 1000
X_train, X_test4, Y_train, Y_test4 = train_test_split(
    X_train, Y_train, test_size=1000, random_state=42)
#fifth 1000
X_train, X_test5, Y_train, Y_test5 = train_test_split(
    X_train, Y_train, test_size=1000, random_state=42)
#six 1000
X_train, X_test6, Y_train, Y_test6 = train_test_split(
    X_train, Y_train, test_size=1000, random_state=42)
#seven 1000
X_train, X_test7, Y_train, Y_test7 = train_test_split(
    X_train, Y_train, test_size=1000, random_state=42)
#eight 1000
X_train, X_test8, Y_train, Y_test8 = train_test_split(
    X_train, Y_train, test_size=1000, random_state=42)
#nine 1000
X_train, X_test9, Y_train, Y_test9 = train_test_split(
    X_train, Y_train, test_size=1000, random_state=42)
#ten 1000
X_train, X_test10, Y_train, Y_test10 = train_test_split(
    X_train, Y_train, test_size=1000, random_state=42)


# #### Linear SVM

# In[10]:

var_regressor = SVR(kernel = 'linear')


# In[11]:

var_regressor.fit(X_train,Y_train)


# In[20]:

predict = var_regressor.predict(X_test1)


# In[21]:

fileobj = open('result1.dat','w')
fileobj.write("actual,predicted\n")
for ind, r in enumerate(predict) :
    fileobj.write(str(Y_test1[ind])+','+str(predict[ind])+'\n')
fileobj.close()


# #### Evaluation Metric

# In[22]:

def calc_accuracy(actual,predicted):
    tc = 0
    for ind in range(len(actual)):
        if abs(fabs(actual[ind])-fabs(predicted[ind])) <= 36000:
            tc = tc + 1
    return float(tc)*100/len(actual)


accuracies = []
msle = []
raw_data = pd.read_csv('result1.dat')
calc = calc_accuracy(raw_data.actual,raw_data.predicted)
#mean_squared_log_error (raw_data.actual,raw_data.actual)    
#msle.append(mean_squared_log_error (raw_data.actual,raw_data.predicted.abs()))
accuracies.append (calc)
print "Accuracy is ",
print accuracies
print "Mean Squared Log Error is : ",
print msle


# In[ ]:



