'''
Created on Dec 25, 2016

@author: loapui
'''

from os.path import join

import pandas as pd
from featureEngineer.base import feature_path, train_path, test_path

def feature_selection(feature, select_feature):
    selected = []
    for _feat in feature.columns:
        for name in select_feature:
            if _feat.find(name) >= 0:
                selected.append(_feat)
    selected = list(set(selected))
    
    return feature[selected]

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


def get_all_continus_feature():
    id_train = pd.read_csv(join(train_path, 'subsidy_train.txt'), sep=',', header=None)
    id_train.columns = ['id', 'money']
    id_train = id_train[['id']]
    id_test = pd.read_csv(join(test_path, 'studentID_test.txt'), sep=',', header=None)
    id_test.columns = ['id']
    
    all_feature = pd.concat([id_train, id_test])
    all_feature = all_feature.drop_duplicates()
    
    card_feature_train = get_card_feature("train")
    card_feature_test = get_card_feature("test")
    card_feature = pd.concat([card_feature_train, card_feature_test])
    all_feature = pd.merge(all_feature, card_feature, how='left', on='id')
    score_feature = get_score_feature()
    all_feature = pd.merge(all_feature, score_feature, how='left', on='id')
    dorm_feature_train = get_dorm_feature('train')
    dorm_feature_test = get_dorm_feature('test')
    dorm_feature = pd.concat([dorm_feature_train, dorm_feature_test])
    all_feature = pd.merge(all_feature, dorm_feature, how='left', on='id')
    
    return all_feature


def get_all_feature(flag):
    
    STUDENTID = {'train' : join(train_path, 'subsidy_train.txt'),
                 'test' : join(test_path, 'studentID_test.txt')}
    
    all_feature = pd.read_csv(STUDENTID[flag], sep=',', header=None)
    if flag == 'train':
        all_feature.columns = ['id', 'money']
    else:
        all_feature.columns = ['id']
    
    card_feature = get_card_feature(flag)
    all_feature = pd.merge(all_feature, card_feature, how='left', on='id')
    score_feature = get_score_feature()
    all_feature = pd.merge(all_feature, score_feature, how='left', on='id')
    dorm_feature = get_dorm_feature(flag)
    all_feature = pd.merge(all_feature, dorm_feature, how='left', on='id')
    id_feature = get_id_feature()
    all_feature = pd.merge(all_feature, id_feature, how='left', on='id')
    zero_feature = get_zero_feture()
    all_feature = pd.merge(all_feature, zero_feature, how='left', on='id')
    ranking_feature = get_ranking_feature()
    all_feature = pd.merge(all_feature, ranking_feature, how='left', on='id')

    all_feature['label'] = 0
    # all_feature.to_csv(join(feature_path, 'all_feature_' + flag + '.csv'), index=False)
    return all_feature


if __name__ == '__main__':
    get_all_feature('train')
    get_all_feature('test')
    