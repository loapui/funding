'''
Created on Dec 25, 2016

@author: loapui
'''

from os.path import join

import pandas as pd
from featureEngineer.base import feature_path, train_path, test_path

def feature_selection(feature, select_feature):
    if len(select_feature) == 0:
        
        return feature
    else:
        selected = []
        for _feat in feature.columns:
            for name in select_feature:
                if _feat.find(name) >= 0:
                    selected.append(_feat)
        selected = list(set(selected))
        
        return feature[selected]


def get_library_feature(flag):
    
    return pd.read_csv(join(feature_path, 'library_features_'+ flag +'.csv'))


def get_zero_feture():
    
    return pd.read_csv(join(feature_path, 'zero_features.csv'))
    

def get_dorm_feature(flag):
    
    return pd.read_csv(join(feature_path, 'dorm_features_'+ flag +'.csv'))


def get_score_feature():
    
    return pd.read_csv(join(feature_path, 'score_features.csv'))


def get_card_feature(flag):
    
    return pd.read_csv(join(feature_path, 'card_features_'+ flag +'.csv')) 


def get_id_feature():
    
    return pd.read_csv(join(feature_path, 'id_features.csv'))


def get_ranking_feature():
    
    return pd.read_csv(join(feature_path, 'RANKING_features.csv'))


def get_all_continus_feature(flag):
    
    if flag == 'train':
        ids = pd.read_csv(join(train_path, 'subsidy_train.txt'), sep=',', header=None)
        ids.columns = ['id', 'money']
        ids = ids[['id']]
    else:
        ids = pd.read_csv(join(test_path, 'studentID_test.txt'), sep=',', header=None)
        ids.columns = ['id']
    
    all_feature = ids
    all_feature = all_feature.drop_duplicates()
    
    # card
    card_feature = get_card_feature(flag)
    all_feature = pd.merge(all_feature, card_feature, how='left', on='id')
    
    # score
    score_feature = get_score_feature()
    all_feature = pd.merge(all_feature, score_feature, how='left', on='id')
    
    # dorm
    dorm_feature = get_dorm_feature(flag)
    all_feature = pd.merge(all_feature, dorm_feature, how='left', on='id')
    
    # library
    #library_feature = get_library_feature(flag)
    #all_feature = pd.merge(all_feature, library_feature, how='left', on='id')
    
    return all_feature


def get_all_feature(flag):
    
    STUDENTID = {'train' : join(train_path, 'subsidy_train.txt'),
                 'test' : join(test_path, 'studentID_test.txt')}
    
    all_feature = pd.read_csv(STUDENTID[flag], sep=',', header=None)
    if flag == 'train':
        all_feature.columns = ['id', 'money']
    else:
        all_feature.columns = ['id']
    
    # card
    card_feature = get_card_feature(flag)
    all_feature = pd.merge(all_feature, card_feature, how='left', on='id')
    
    # score
    score_feature = get_score_feature()
    all_feature = pd.merge(all_feature, score_feature, how='left', on='id')
    
    # dorm
    dorm_feature = get_dorm_feature(flag)
    all_feature = pd.merge(all_feature, dorm_feature, how='left', on='id')
    
    # library
    library_feature = get_library_feature(flag)
    all_feature = pd.merge(all_feature, library_feature, how='left', on='id')
    
    # id
    id_feature = get_id_feature()
    all_feature = pd.merge(all_feature, id_feature, how='left', on='id')
    
    # zero
    zero_feature = get_zero_feture()
    all_feature = pd.merge(all_feature, zero_feature, how='left', on='id')
    
    # ranking
    ranking_feature = get_ranking_feature()
    all_feature = pd.merge(all_feature, ranking_feature, how='left', on='id')

    all_feature['label'] = 0
    # all_feature.to_csv(join(feature_path, 'all_feature_' + flag + '.csv'), index=False)
    return all_feature
    