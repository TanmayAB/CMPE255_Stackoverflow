
# coding: utf-8

# In[125]:


import matplotlib.pyplot as plt
from math import fabs
import pandas as pd
from sklearn.metrics import f1_score,accuracy_score, mean_squared_error, mean_squared_log_error


# In[126]:


def calc_accuracy(actual,predicted):
    tc = 0
    for ind in range(len(actual)):
        if abs(fabs(actual[ind])-fabs(predicted[ind])) <= 36000:
            tc = tc + 1
    return float(tc)*100/len(actual)


# In[127]:


accuracies = []
msle = []
for i in range(10) :
    raw_data = pd.read_csv('./CMPE255_Stackoverflow/Result/PassiveAggressiveResult/result'+str(i+1)+'.dat')
    calc = calc_accuracy(raw_data.actual,raw_data.predicted)
    mean_squared_log_error(raw_data.actual,raw_data.actual)    
    msle.append(mean_squared_log_error (raw_data.actual,raw_data.predicted.abs()))
    accuracies.append (calc)
print "Accuracy is ",
print accuracies
print "Mean Squared Log Error is : ",
print msle


# In[128]:


x = [1,2,3,4,5,6,7,8,9,10]

plt.bar(x,accuracies,align='center')
plt.ylabel('Accuracy(10 hours)')
plt.xlabel('10 folds')
plt.show()

plt.bar(x, msle, align = 'center')
plt.ylabel('Mean squared log error')
plt.xlabel('10 folds')
plt.show()


# In[119]:


print "Average Accuracy : ",
print sum(accuracies)/len(accuracies)
print "Average Mean Squared Error : ",
print sum(msle)/len(msle)


# In[98]:


import numpy as np
import matplotlib.pyplot as plt
 
# data to plot
n_groups = 3

#Individual obtained accuracy and mean squared log errors values of KNN, SVR and PassiveAggressive regression respectively and used below for generating a graph
accuracy = (65.2, 79.65, 80.50)
mean_squared_log_error = (6.55, 7.93, 6.99)


# create plot
fig, ax = plt.subplots(figsize=(9,6))
index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.8
 
rects1 = plt.bar(index, accuracy, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Accuracy')


rects2 = plt.bar(index + bar_width, mean_squared_log_error, bar_width,
                 alpha=opacity,
                 color='y',
                 label='Mean-squared-log-error')

plt.xlabel('Parameters')
plt.ylabel('Percentage')
plt.title('Performance Graph')
plt.xticks(index + bar_width, ('KNN Regressor', 'Linear SVR', 'Passive Aggressive Regressor'))
plt.legend(loc = 'upper right')
 
plt.tight_layout()
plt.show()

