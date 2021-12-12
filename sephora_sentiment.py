#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 23:00:10 2021

@author: alicehuang
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn                         import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors               import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 6)
pd.set_option('display.width',800)

dta = pd.read_excel('/Users/alicehuang/Desktop/SCHOOL/brandeis/MARKETING ANALYTICS/data files/ordinary-nia.xlsx',index_col=0).reset_index()
dta = dta[['one_review_text','star_rating','skin_type','clean date']]

s_type         = pd.DataFrame(np.sort(np.unique(dta['skin_type'].astype(str)))).reset_index()
s_type.columns = ['skin_index','skin_type']  
dta = pd.merge(dta,s_type,on = "skin_type",how ="left")

dta.head()
s_type

#%%% define functions
'''best K function'''
def Best_k(X_train,X_valid,Y_train,Y_valid):
    knn = 0
    best_accuracy = 0
    best_k = 0
    k = 1
    max_k = 100
    y_hat_valid = []
    for k in range(1,max_k+1):
        clf = KNeighborsClassifier(n_neighbors=k).fit(X_train , Y_train )
        y_hat_valid.append(np.concatenate([clf.predict(X_valid)]))
    for knn in range(max_k-1):
        conf_matrix = (confusion_matrix(Y_valid , y_hat_valid[knn]))
        correct =  np.trace(conf_matrix)
        total = len(Y_valid)
        accuracy_rate = (correct/total)
        if accuracy_rate > best_accuracy:
            best_accuracy = accuracy_rate
            best_k = knn+1
    print('The best k is',best_k,'Accuracy:', round(best_accuracy,2))
    return best_k

'''prediction'''
def Prediction(best_k,X_train,Y_train,X_valid,X_test):
    Y_hat_train = []
    Y_hat_valid = []
    Y_hat_test  = []
    clf = KNeighborsClassifier(n_neighbors=best_k).fit(X_train , Y_train )
    Y_hat_train.append(clf.predict(X_train))
    Y_hat_valid.append(clf.predict(X_valid))
    Y_hat_test.append(clf.predict(X_test))
    
    Y_hat_train = pd.DataFrame(Y_hat_train).transpose()
    Y_hat_valid = pd.DataFrame(Y_hat_valid).transpose()
    Y_hat_test = pd.DataFrame(Y_hat_test).transpose()
    
    Y_hat_train.columns = ['label_hat']
    Y_hat_valid.columns = ['label_hat']
    Y_hat_test.columns = ['label_hat']

    return ({'Y_hat_train':Y_hat_train, 'Y_hat_valid': Y_hat_valid, 'Y_hat_test':Y_hat_test})

'''logistic regression'''
def logistic_reg_classifier_mult_labels(X_train,Y_train,X_valid,Y_valid,X_test,Y_test):
    
    categories         = pd.DataFrame(np.sort(np.unique(Y_train))).reset_index()
    categories.columns = ['index','label']   

    ccp_train_list = []
    ccp_valid_list = []
    ccp_test_list  = []
    for cat in categories['label'].to_list():
        Y_train_c = 1*(Y_train==cat)
        clf       = linear_model.LogisticRegression(tol          = 0.0001,
                                                    max_iter     = 10000,
                                                    random_state = None).fit(X_train, Y_train_c)
        ccp_train_list.append(  clf.predict_proba(X_train)[:,1])
        ccp_valid_list.append(  clf.predict_proba(X_valid)[:,1])
        ccp_test_list.append(   clf.predict_proba(X_test)[:,1])
    
    ' . Topic probability matrix'
    ccp_train = pd.DataFrame(ccp_train_list).transpose()
    ccp_valid = pd.DataFrame(ccp_valid_list).transpose()
    ccp_test  = pd.DataFrame(ccp_test_list).transpose()

    ' . Choosing your predictive category for the y'
    ccp_train['index_hat'] =  ccp_train.idxmax(axis=1)
    ccp_valid['index_hat'] =  ccp_valid.idxmax(axis=1)
    ccp_test[ 'index_hat'] =  ccp_test.idxmax(axis=1)
    
    ccp_train              = ccp_train.merge(categories, 
                                              left_on  = 'index_hat' ,
                                              right_on = 'index'     , 
                                              how      = 'left').rename(columns={'label':'label_hat'}).drop(['index','index_hat'],axis=1)
    ccp_valid              = ccp_valid.merge(categories,
                                              left_on    = 'index_hat',
                                              right_on   = 'index', 
                                              how        = 'left').rename(columns={'label':'label_hat'}).drop(['index','index_hat'],axis=1)
    ccp_test               = ccp_test.merge(categories,
                                            left_on   = 'index_hat' ,
                                            right_on  = 'index' ,
                                            how       = 'left').rename(columns={'label':
                                                                                'label_hat'}).drop(['index','index_hat'],axis=1)  
    ccp_train['Y_train']   = Y_train
    ccp_valid['Y_valid']   = Y_valid
    ccp_test['Y_test']     = Y_test
    return({'ccp_train'  : ccp_train,'ccp_valid'  : ccp_valid,'ccp_test'   : ccp_test})

'''confusion matrix function'''
def Conf_Matrix(Y_test,result):
    conf_matrix = (confusion_matrix(Y_test , result))
    correct =  np.trace(conf_matrix)
    total = conf_matrix.sum()
    accuracy_rate = (correct/total)
    return print(conf_matrix,'Accuracy:', round(accuracy_rate,2))

'''tfidf'''
def graph_tfidf (ngram_range,max_df,min_df,title,corpus):
    vectorizer_tf_idf = TfidfVectorizer(lowercase   = True,
                                        ngram_range = ngram_range,
                                        max_df      = max_df,
                                        min_df      = min_df,
                                        stop_words='english');
    
    X_tf       = vectorizer_tf_idf.fit_transform(corpus)
    columns = vectorizer_tf_idf.get_feature_names()
    df = pd.DataFrame(X_tf.toarray(), columns = columns,)
    df1 = df.sum(axis = 0 ).reset_index()
    df1.columns = ['Words','Weight']
    df2 = df1.sort_values(['Weight'], ascending=False).head(15)
    
    #graph
    plt.barh(df2['Words'],df2['Weight'], color = 'coral')
    plt.xlabel('Weight')
    plt.gca().invert_yaxis()
    plt.title(title)
    return plt.show()


#%%% TFIDF graph

#high reviews
dta_high = dta[dta['star_rating']>=3]
corpus = dta_high['one_review_text'].to_numpy()

ngram_range = (2,3)
max_df = 0.95
min_df = 0.01

graph_tfidf (ngram_range,max_df,min_df,corpus)
title = 'Term Frequency in High Rating Reviews'

#low reviews
dta_low = dta[dta['star_rating']<3]
corpus = dta_low['one_review_text'].to_numpy()

ngram_range = (3,4)
max_df = 0.95
min_df = 0.01
title = 'Term Frequency in Low Rating Reviews'

graph_tfidf (ngram_range,max_df,min_df,corpus)

#%%% Corpus

corpus = dta['one_review_text'].to_numpy()

ngram_range = (1,1)
max_df = 0.85
min_df = 0.01
vectorizer = CountVectorizer(lowercase   = True,
                                  ngram_range = ngram_range,
                                  max_df      = max_df     ,
                                  min_df      = min_df     );
                                  
X = vectorizer.fit_transform(corpus)
X.toarray()

#%%% Split Data
np.random.seed(1)
dta['ML_group']     = np.random.randint(100,size = dta.shape[0])
dta                 = dta.sort_values(by='ML_group')
index_train         = dta.ML_group<80                     
index_valid         = (dta.ML_group>=80)&(dta.ML_group<90)
index_test          = (dta.ML_group>=90)


#%%%
#performing the TVT - SPLIT
Y_train   = dta.star_rating[index_train].to_numpy()
Y_valid   = dta.star_rating[index_valid].to_numpy()
Y_test    = dta.star_rating[index_test ].to_numpy()

X_train   = X[np.where(index_train)[0],:]
X_valid   = X[np.where(index_valid)[0],:]
X_test    = X[np.where(index_test) [0],:]

#%%% reviews predict star rating

'''KNN'''

#find best k
best_k = Best_k(X_train,X_valid,Y_train,Y_valid)

#generate prediction
KNN_Pred = Prediction(best_k,X_train,Y_train,X_valid,X_test)
Conf_Matrix(Y_test,KNN_Pred['Y_hat_test']['label_hat'])

'''Logistic'''
logistic_results = logistic_reg_classifier_mult_labels(X_train,Y_train,X_valid,Y_valid,X_test,Y_test)
Conf_Matrix(Y_test,logistic_results['ccp_test']['label_hat'])

#%%% reviews predict skin type

''' KNN'''
Y_train   = dta.skin_index[index_train].to_numpy()
Y_valid   = dta.skin_index[index_valid].to_numpy()
Y_test    = dta.skin_index[index_test ].to_numpy()

#find best k
best_k = Best_k(X_train,X_valid,Y_train,Y_valid)

#generate prediction
KNN_Pred = Prediction(best_k,X_train,Y_train,X_valid,X_test)
Conf_Matrix(Y_test,KNN_Pred['Y_hat_test']['label_hat'])

'''Logistics'''
Y_train   = dta.skin_type[index_train].to_numpy()
Y_valid   = dta.skin_type[index_valid].to_numpy()
Y_test    = dta.skin_type[index_test ].to_numpy()

logistic_results = logistic_reg_classifier_mult_labels(X_train,Y_train,X_valid,Y_valid,X_test,Y_test)
Conf_Matrix(Y_test,logistic_results['ccp_test']['label_hat'])

#%%% skin type predict star rating

#average rating by skin type for high and low reviews
high_skin = dta_high.groupby(['skin_type']).agg({"star_rating" : np.mean})
low_skin = dta_low.groupby(['skin_type']).agg({"star_rating" : np.mean})

#convert X variable to dummy variables
X1 = pd.get_dummies(dta['skin_type']).to_numpy()

#split data
Y_train   = dta.star_rating[index_train].to_numpy()
Y_valid   = dta.star_rating[index_valid].to_numpy()
Y_test    = dta.star_rating[index_test ].to_numpy()

X_train   = X1[np.where(index_train)[0],:]
X_valid   = X1[np.where(index_valid)[0],:]
X_test    = X1[np.where(index_test)[0],:]

''' KNN'''
#find best k
best_k = Best_k(X_train,X_valid,Y_train,Y_valid)

#generate prediction
KNN_Pred = Prediction(best_k,X_train,Y_train,X_valid,X_test)
Conf_Matrix(Y_test,KNN_Pred['Y_hat_test']['label_hat'])

'''Logistics'''

X1 = pd.get_dummies(dta['skin_type'],drop_first=True).to_numpy()

X_train   = X1[np.where(index_train)[0],:]
X_valid   = X1[np.where(index_valid)[0],:]
X_test    = X1[np.where(index_test)[0],:]

logistic_results = logistic_reg_classifier_mult_labels(X_train,Y_train,X_valid,Y_valid,X_test,Y_test)
Conf_Matrix(Y_test,logistic_results['ccp_test']['label_hat'])

