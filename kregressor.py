from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_log_error
from matplotlib import pyplot as plt

# reading data
rawdata = pd.read_csv('C:/Users/gaura/Desktop/songs/StackOverflow2million/StackOverflow2million.csv')

# attaching tags to questions with weight = 2 for tags
title_tags = rawdata[:]
title_tags.columns = title_tags.columns.str.lower()
title_tags.questions_tags = title_tags.questions_tags.str.replace('|', ' ')
title_tags.questions_tags2 = title_tags.questions_tags.str.cat(title_tags.questions_tags, sep = ' ')
title_tags.questions_title = title_tags.questions_title.str.cat(title_tags.questions_tags2, sep = ' ')

# pass analyser to CountVectorizer
wnl = WordNetLemmatizer()
analyzer = CountVectorizer().build_analyzer()

def lemmatized_words(doc):
    return (wnl.lemmatize(w) for w in analyzer(doc))

vectorizer = CountVectorizer(analyzer=lemmatized_words, stop_words='english')


# generates train and test matrix based on the index provided
def traintest(test1, test2):
    testdata = title_tags[test1: test2]
    #     print testdata.head()
    drop = pd.Series(range(test1, test2))
    #     print drop
    traindata = title_tags.drop(title_tags.index[[drop]])
    #     traindata = title_tags[:]
    #     traindata = traindata.drop(traindata.index[[test1,test2]])
    #     print traindata.head()
    train_xformed = vectorizer.fit_transform(traindata.questions_title)
    print train_xformed.shape
    test_xformed = vectorizer.transform(testdata.questions_title)
    print test_xformed.shape
    time = traindata.time
    time = time.astype('float64')
    return (train_xformed, test_xformed, time)


# function responsible for prediction after generation of train, test matrices
def predictor(ind1, ind2):
    ans = []
    split = traintest(ind1, ind2)
    regressor = KNeighborsRegressor(n_neighbors=3, weights='distance')
    regressor = regressor.fit(split[0], split[2])
    iterations = int((ind2 - ind1) / 200)
    i1 = 0
    neigh = [0] * 200
    dist = [0] * 200
    for x in range(iterations):
        test = split[1]
        y = regressor.predict(test[i1:(i1 + 200)])
        ans.extend(y)
        i1 = i1 + 200
    return ans


# testing 10 sets of 1000 records(taken from 10 sets of 200000 records)
ans1 = predictor(1950000,1951000)
ans2 = predictor(150000,151000)
ans3 = predictor(350000,351000)
ans4 = predictor(550000,551000)
ans5 = predictor(750000,751000)
ans6 = predictor(950000,951000)
ans7 = predictor(1150000,1151000)
ans8 = predictor(1350000,1351000)
ans9 = predictor(1550000,1551000)
ans10 = predictor(1750000,1751000)


# function for calculating accuracy
def calc_accuracy(arr, ind1, ind2):
    tc = 0
    timetest = title_tags[ind1:ind2].time.astype('float64')
    for x, y in np.nditer([arr, timetest]):
        if abs(x - y) <= 36000:
            tc = tc + 1
    return tc


# function for calculating mean_squared_log_error
def calc_meanlog(arr, ind1, ind2):
    timetest = title_tags[ind1:ind2].time.astype('float64')
    l1 = []
    l2 = []

    for x in range(1000):
        l1.append(arr[x] / 3600)

    for x in range(1000):
        l2.append(timetest[ind1 + x] / 3600)

    return mean_squared_log_error(l2, l1)


acc1 = calc_accuracy(ans1,1950000,1951000)
acc2 = calc_accuracy(ans2,150000,151000)
acc3 = calc_accuracy(ans3,350000,351000)
acc4 = calc_accuracy(ans4,550000,551000)
acc5 = calc_accuracy(ans5,750000,751000)
acc6 = calc_accuracy(ans6,950000,951000)
acc7 = calc_accuracy(ans7,1150000,1151000)
acc8 = calc_accuracy(ans8,1350000,1351000)
acc9 = calc_accuracy(ans9,1550000,1551000)
acc10 = calc_accuracy(ans10,1750000,1751000)


acc1 = (float)(acc1*100/1000)
acc2 = (float)(acc2*100/1000)
acc3 = (float)(acc3*100/1000)
acc4 = (float)(acc4*100/1000)
acc5 = (float)(acc5*100/1000)
acc6 = (float)(acc6*100/1000)
acc7 = (float)(acc7*100/1000)
acc8 = (float)(acc8*100/1000)
acc9 = (float)(acc9*100/1000)
acc10 = (float)(acc10*100/1000)


#  finding mean accuracy
acc = acc1+acc2+acc3+acc4+acc5+acc6+acc7+acc8+acc9+acc10
print (float(acc/10))

log1 = calc_meanlog(ans1,1950000,1951000)
log2 = calc_meanlog(ans2,150000,151000)
log3 = calc_meanlog(ans3,350000,351000)
log4 = calc_meanlog(ans4,550000,551000)
log5 = calc_meanlog(ans5,750000,751000)
log6 = calc_meanlog(ans6,950000,951000)
log7 = calc_meanlog(ans7,1150000,1151000)
log8 = calc_meanlog(ans8,1350000,1351000)
log9 = calc_meanlog(ans9,1550000,1551000)
log10 = calc_meanlog(ans10,1750000,1751000)

# finding mean error
log = log1 + log2 + log3 + log4 + log5 + log6 + log7 + log8 + log9 + log10
print (float(log/10))

x = [1,2,3,4,5,6,7,8,9,10]
y = [acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10]
y2 = [log1, log2, log3, log4, log5, log6, log7, log8, log9, log10]

# **********************************uncomment the following to get graph plots******************************************
# # plotting accuracy
# plt.bar(x,y,align='center')
# plt.ylabel('Accuracy(10 hours)')
# plt.xlabel('10 folds')
# plt.show()
#
# # plotting mean squared log error
# plt.bar(x, y2, align = 'center')
# plt.ylabel('Mean squared log error')
# plt.xlabel('10 folds')
# plt.show()