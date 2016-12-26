'''
Created on Dec 25, 2016

@author: loapui
'''

from collections import Counter

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from featureEngineer.base import sample
from featureEngineer.features import feature_selection
from featureEngineer.feature_extraction import extract_ranking_lda_dynamic


_PIPELINE = {'0vsRest' : {'feature' : ["pos", "balance", "payget", "recharge", "lost", "rank", "academy", \
                                       "id_", "RANKING_lda", "nZero"],
                          'sample' : {'type' : 'under_sample', 'ratio' : 1.0}},
             
             '1vs23' : {'feature' : ["pos", "balance", "rank", "academy", "RANKING_lda"],
                        'sample' : {'type' : 'under_sample', 'ratio' : 1.0}},
              
             '2vs3' : {'feature' : ["pos", "balance", "rank", "academy", "RANKING_lda"],
                       'sample' : {'type' : 'over_sample', 'ratio' : 1.0}} 
             }

def _train_feature_analyse(train_feature):
    y = train_feature['money']
    counter = Counter(y)
    print 'trainY distribution: ', counter


def _train_and_pred_util(train_feature, test_feature, target, select_feature, sample_params):
    trainY = train_feature['label']
    trainX = feature_selection(train_feature, select_feature)
    testX_id = test_feature['id'].values
    testX = feature_selection(test_feature, select_feature)
    counter = Counter(trainY)
    print '# before sample: positive: %d, negative: %d' % (counter[1], counter[0])
    
    # random sample
    X_sample, y_sample = sample(trainX, trainY, \
                                sample_params = sample_params)
    counter = Counter(y_sample)
    print '# after sample: positive: %d, negative: %d' % (counter[1], counter[0])
    
    # model
    print "# train.shape: ", X_sample.shape
    clf = RandomForestClassifier(n_estimators=500, n_jobs=20, random_state=1)
    clf = clf.fit(X_sample, y_sample)
    result = clf.predict(testX)
    
    test_result = pd.DataFrame(columns=["studentid", "subsidy"])
    test_result.studentid = testX_id
    test_result.subsidy = result
    test_result.subsidy = test_result['subsidy'].apply(lambda x:int(x))
    
    return test_result[test_result.subsidy == target], \
        test_feature[test_feature['id'].isin(test_result['studentid'][test_result.subsidy != target])]
    

def _train_and_pred_2vs3(train_feature, test_feature, pipeline=_PIPELINE):
    money_to_label = {0 : -1,
                      1000 : -1,
                      1500 : 0,
                      2000 : 1}
    
    train_feature.label = train_feature['money'].map(money_to_label)
    train_feature = train_feature[train_feature.label >= 0]
    
    studentID_label2, test_feature = _train_and_pred_util(train_feature, test_feature, \
                                                        target = 0, \
                                                        select_feature = pipeline['2vs3']['feature'],\
                                                        sample_params = pipeline['2vs3']['sample'])
    studentID_label2['subsidy'] = studentID_label2['subsidy'].map({0 : 1500})
    
    return studentID_label2, test_feature
    

def _train_and_pred_1vs23(train_feature, test_feature, pipeline=_PIPELINE):
    money_to_label = {0 : -1,
                      1000 : 1,
                      1500 : 0,
                      2000 : 0}
    
    train_feature.label = train_feature['money'].map(money_to_label)
    train_feature = train_feature[train_feature.label >= 0]
    
    studentID_label1, test_feature = _train_and_pred_util(train_feature, test_feature, \
                                                        target = 1, \
                                                        select_feature = pipeline['1vs23']['feature'],\
                                                        sample_params = pipeline['1vs23']['sample'])
    studentID_label1['subsidy'] = studentID_label1['subsidy'].map({1 : 1000})
    
    return studentID_label1, test_feature


def _train_and_pred_0vsRest(train_feature, test_feature, pipeline=_PIPELINE):
    money_to_label = {0 : 0,
                      1000 : 1,
                      1500 : 1,
                      2000 : 1}
    
    train_feature.label = train_feature['money'].map(money_to_label)
    studentID_label0, test_feature = _train_and_pred_util(train_feature, test_feature, \
                                                        target = 0, \
                                                        select_feature = pipeline['0vsRest']['feature'], \
                                                        sample_params = pipeline['0vsRest']['sample'])
    
    return studentID_label0, test_feature
    

def cascadedClassifier(train_feature, test_feature):
    # extract lda feature
    train_feature, test_feature = extract_ranking_lda_dynamic(train_feature, test_feature)
    
    studentID_label = _train_and_pred(train_feature, test_feature)
    studentID_label = studentID_label.sort_values(by='studentid', ascending=True)
    
    _train_feature_analyse(train_feature)

    return studentID_label
    
    
def _train_and_pred(train_feature, test_feature):
    print "0vsRest..."
    studentID_label0, test_feature = _train_and_pred_0vsRest(train_feature, test_feature)
    print "1vs23..."
    studentID_label1, test_feature = _train_and_pred_1vs23(train_feature, test_feature)
    print "2vs3..."
    studentID_label2, test_feature = _train_and_pred_2vs3(train_feature, test_feature)
    
    studentID_label3 = pd.DataFrame(columns = ["studentid", "subsidy"])
    studentID_label3.studentid = test_feature['id'].values
    studentID_label3.subsidy = [2000] * len(studentID_label3.studentid)
    
    studentID_label = pd.concat([studentID_label0, studentID_label1, studentID_label2, studentID_label3])
    print "testY distribution: ", Counter(studentID_label.subsidy)
    
    return studentID_label