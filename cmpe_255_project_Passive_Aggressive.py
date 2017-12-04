
# coding: utf-8

# In[353]:


import re
import string
import pandas as pd
from math import fabs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression,LinearRegression,PassiveAggressiveRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from nltk.stem.porter import PorterStemmer
from scipy.sparse import hstack
from sklearn.svm import SVR
import nltk
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import f1_score,accuracy_score
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.model_selection import KFold


# In[ ]:


nltk.download()


# In[386]:


stop = set(stopwords.words('english'))
raw_data = pd.read_csv('StackOverflow2million.csv')


# # Preprocessing total data

# ### Splitting Question titles

# In[387]:



#remove punctuations from a string and convert to lower case
raw_data['questions_title'] = raw_data['questions_title'].apply(lambda x:''.join([i.lower() for i in x 
                                                  if i not in string.punctuation]))

#remove stop words
raw_data['questions_title'] = raw_data['questions_title'].apply(lambda x:' '.join([i for i in x.split(' ') 
                                                  if i not in stop]))

#removing digits
raw_data['questions_title'] = raw_data['questions_title'].apply(lambda x:' '.join([i.replace(r'[0-9]+','') for i in x.split(' ') ]))


# ### Splitting Question Tags

# In[388]:


#remove pipe and seperate tags
raw_data['questions_tags'] = raw_data['questions_tags'].apply(lambda x:' '.join([i.lower() for i in x.split('|')]))


# ## Vectorizing the data and performing stemming

# In[389]:


stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    # remove non letters
    text = re.sub("[^a-zA-Z]", " ", text)
    # tokenize
#     tokens = nltk.word_tokenize(text)
    tokens = text.split()
    # stem
    stems = stem_tokens(tokens, stemmer)
    return stems

vectorizer = CountVectorizer(
    analyzer='word',
    tokenizer=tokenize,
    lowercase=True,
    stop_words='english'
)
train_x_que_tag = vectorizer.fit_transform(raw_data.questions_tags)
train_x_que_title = vectorizer.fit_transform(raw_data.questions_title)
train_y = raw_data['time'].tolist()


# In[390]:


print train_x_que_tag.shape
print train_x_que_title.shape


# In[391]:


train_x =  hstack((train_x_que_tag, train_x_que_title))


# In[392]:


print train_x.shape


# In[364]:


# X_train1, X_test1, Y_train1, Y_test1 = train_test_split(
#     train_x, train_y, test_size=2000)
# X_train2, X_test2, Y_train2, Y_test2 = train_test_split(
#     train_x, train_y, test_size=2000)
# X_train3, X_test3, Y_train3, Y_test3 = train_test_split(
#     train_x, train_y, test_size=2000)
# X_train4, X_test4, Y_train4, Y_test4 = train_test_split(
#     train_x, train_y, test_size=2000)
# X_train5, X_test5, Y_train5, Y_test5 = train_test_split(
#     train_x, train_y, test_size=2000)
# X_train6, X_test6, Y_train6, Y_test6 = train_test_split(
#     train_x, train_y, test_size=2000)
# X_train7, X_test7, Y_train7, Y_test7 = train_test_split(
#     train_x, train_y, test_size=2000)
# X_train8, X_test8, Y_train8, Y_test8 = train_test_split(
#     train_x, train_y, test_size=2000)
# X_train9, X_test9, Y_train9, Y_test9 = train_test_split(
#     train_x, train_y, test_size=2000)
# X_train10, X_test10, Y_train10, Y_test10 = train_test_split(
#     train_x, train_y, test_size=200)


# ## PassiveAggressive Regression

# In[393]:


def regressor(X_train,Y_train,X_test):
    Linear_Regressor = PassiveAggressiveRegressor()
    Linear_Regressor.fit(X_train,Y_train)
    prediction_result = Linear_Regressor.predict(X_test)
    return prediction_result


# In[394]:


print result


# In[ ]:


accuracies = []
for i in range(0,10):
    X_train, X_test, Y_train, Y_test = train_test_split(
    train_x, train_y, test_size=2000)
    result = regressor(X_train,Y_train,X_test)
    validate_result(Y_test,result)
    accuracies.append(calc_accuracy(Y_test,result))
    #writetofile(Y_test,result,i)
print accuracies


# ## Evaluation Metrics

# In[323]:


def calculate_variance(Y_test,result):
    variance = explained_variance_score(Y_test, result, sample_weight=None,multioutput='uniform_average')
    print "variance (best 2) : " + str(variance)


# In[314]:


def calculate_accuracy(Y_test,result):
    actual_values = []
    predicted_values = []
    for i in range(0,len(Y_test)):
        actual_values.append(Y_test[i])
        predicted_values.append(result[i]/)
    actual_labels = []

    prediction_labels =  []
    for x in range(len(actual_values)):
        diff=math.fabs(float(predicted_values[x])-float(actual_values[x]))
        if float(actual_values[x])!=0 and (diff/float(actual_values[x]))<0.30:
            prediction_labels.append(1)
        else:
            prediction_labels.append(0)
        actual_labels.append(1)
    accuracy= accuracy_score(actual_labels,prediction_labels)
    print "Accuracy (best 1) : "+ str(accuracy)


# In[320]:


def validate_result(Y_test,result):
    calculate_variance(Y_test,result)
    calculate_accuracy(Y_test,result)


# In[330]:


def writetofile(Y_test,result,i):
    f = open("result" + str(i+1),"w")
    for j in range(0,len(Y_test)):
        f.write(str(Y_test[j]) + "," + str(result[j]) + "\n")
    f.close()


# In[370]:


def calc_accuracy(actual,predicted):
    tc = 0
    for ind in range(len(actual)):
        if abs(fabs(actual[ind])-fabs(predicted[ind])) <= 36000:
            tc = tc + 1
    return float(tc)*100/len(actual)


# In[383]:


accuracy = []
for i in range(10) :
    raw_data = pd.read_csv('./result'+str(i+1) + '.dat')
    calc = calc_accuracy(raw_data.actual,raw_data.predicted)
    accuracy.append (calc)


# In[384]:


print accuracy

