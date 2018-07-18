#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 18:03:56 2018

@author: vikramreddy
"""
#loading requied libraries
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import ADASYN
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#loading data
train=pd.read_csv('Train_data.csv')
test=pd.read_csv('Test_data.csv')

#check for missing values
train.describe()
train.isnull().values.ravel().sum()
#correlation plot of predictor variables
cor=train.iloc[:,0:18]
sns.heatmap(cor.corr(),cmap=sns.diverging_palette(220, 10, as_cmap=True))



# creating function for data preprocessing
def data_cleaning(dat):
    #removing the variables which has high correlation

    dat=dat.drop(['number vmail messages', 'total day charge',
                  'total night charge','total intl charge'], axis=1)

    #data pre processing on 
    #removing columns which are not useful
    dat=dat.drop(['state', 'phone number'], axis=1)
    dat['international plan'] = pd.factorize(dat['international plan'])[0]
    dat['voice mail plan'] = pd.factorize(dat['voice mail plan'])[0]
    dat['Churn'] = pd.factorize(dat['Churn'])[0]
    return dat


#calling data_cleaning
train=data_cleaning(train) 
test=data_cleaning(test)
#imbalance check in output
train['Churn'].hist()



#outlier detection
plt.boxplot(train)
columns=['account length','total day minutes','total day calls','total eve minutes','total eve calls',
         'total eve charge','total night minutes','total night calls','total intl minutes',
         'total intl calls']
down=train.quantile(0.05)
up=train.quantile(0.95)
pd.options.mode.chained_assignment = None
data_without_outlier=train
for column in columns:
    data_without_outlier[data_without_outlier[column]<down[column]][column]=down[column]
    data_without_outlier[data_without_outlier[column]>up[column]][column]=up[column]
    
    
    
#over sampling for data with outliers
sm = ADASYN()
x=pd.DataFrame(train.iloc[:,0:14])
y=pd.DataFrame(train.iloc[:,14])
x1,y1 = sm.fit_sample(x,y.values.ravel())
train=pd.DataFrame(x1,columns=list(x))
train['Churn']=y1
#checking class distribution after applyning over sampling
train['Churn'].hist()
#shuffling data
train=shuffle(train)
test=shuffle(test)




#oversampling of data with out outliers
sm_clean = ADASYN()
x_clean=pd.DataFrame(data_without_outlier.iloc[:,0:14])
y_clean=pd.DataFrame(data_without_outlier.iloc[:,14])
x2,y2 = sm_clean.fit_sample(x_clean,y_clean.values.ravel())
clean_train=pd.DataFrame(x1,columns=list(x_clean))
clean_train['Churn']=y2
#checking class distribution after applyning over sampling for data without outliers
clean_train['Churn'].hist()
#shuffling data
clean_train=shuffle(clean_train)
test=shuffle(test)

#creating a function for cross validation
def cross_valid(model,test,train):
    model.fit(train.iloc[:,0:14],train['Churn'])
    #prediction of model with out using cross validation
    pred=model.predict(test.iloc[:,0:14])
    print("confusion matrix  of predictions(not probability predictions) and model with out using cross validation")
    print(confusion_matrix(test['Churn'], pred))
    #predictions using cross validation
    predicted = pd.DataFrame(cross_val_predict(model, test.iloc[:,0:14],test['Churn'],cv=10))
    print("accuracy of cross validation predictions(not probability predition)")
    print(metrics.accuracy_score(test['Churn'], predicted))
    #finding the score for each cross validation
    accuracy = cross_val_score(model, test.iloc[:,0:14],test['Churn'], cv=10,scoring='roc_auc')
    print("mean accuracy score of  10 cross validations")
    print (accuracy.mean())
    #predicting probabilities from cross validation
    predicted_proba = pd.DataFrame(cross_val_predict(model, test.iloc[:,0:14],test['Churn'],cv=10,method='predict_proba'))
    print("predicted probabilities from cross validation")
    print(predicted_proba)
    fpr, tpr, thresholds =roc_curve(test['Churn'], predicted_proba.iloc[:,1])
    roc_auc = auc(fpr, tpr)
    print("area under the curve")
    print(roc_auc)
    #determing threshold
    # The optimal cut off would be where tpr is high and fpr is low
    # tpr - (1-fpr) is zero or near to zero is the optimal cut off point
    i = np.arange(len(tpr)) # index for df
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
    thresh=roc.iloc[(roc.tf-0).abs().argsort()[0:1]]
    threshold=thresh[['thresholds']]
    print("threshold obtained from ROC curve")
    print(threshold.iloc[0,0])
    #mapping threshlod to the predicted probabilities
    output= predicted_proba.iloc[:,1].map(lambda x:1  if x > threshold.iloc[0,0] else 0)
    print("confusion matrix for predictions(after applying threshold to probabilities) and test data ")
    print(confusion_matrix(test['Churn'], output))
    print("accuracy of cross validation predictions( probability prediction)")

    print(metrics.accuracy_score(test['Churn'], output))
    #plotting Roc curve
    fig, ax = plt.subplots()
    plt.plot(roc['tpr'])
    plt.plot(roc['1-fpr'], color = 'red')
    plt.xlabel('1-False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    ax.set_xticklabels([])
    return ""

#logistic regression 
logreg=LogisticRegression()
#calling the function for data with outliers
cross_valid(logreg,test,train)
#calling the function for data with out outliers(clean data)
cross_valid(logreg,test,clean_train)



#random forest
rf = RandomForestClassifier(n_estimators=500)
#calling the function for data with outliers
cross_valid(rf,test,train)
#calling the function for data with out outliers(clean data)
cross_valid(rf,test,clean_train)

#naive bayes model
gnb = GaussianNB()
#calling the function for data with outliers
cross_valid(gnb,test,train)
#calling the function for data with out outliers(clean data)
cross_valid(gnb,test,clean_train)

#kNN model
kNN=KNeighborsClassifier(n_neighbors=20, radius=5.0, algorithm='auto')
#calling the function for data with outliers
cross_valid(kNN,test,train)
#calling the function for data with out outliers(clean data)
cross_valid(kNN,test,clean_train)

