#coding=utf-8
'''
Created on Dec 25, 2016

@author: loapui
'''

from os.path import join
from collections import Counter

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from featureEngineer.base import train_path, test_path, feature_path, get_date_info
from features import feature_selection
from featureEngineer.features import get_all_continus_feature

def extract_zero_feature():
    all_feature = get_all_continus_feature()
    #fill na
    all_feature = all_feature.fillna(0)
    
    all_feature['nZero'] = all_feature.apply(lambda r: Counter(r)[0], axis = 1)
    zero_feature = all_feature[['id', 'nZero']]
    
    zero_feature.to_csv(join(feature_path, 'zero_features.csv'), index=False)
    

def extract_dorm_feature(flag):
    dorm_file = {'train' : join(train_path, 'dorm_train.txt'),
             'test' : join(test_path, 'dorm_test.txt')}
    dorm_records = pd.read_csv(dorm_file[flag], sep=',', header=None)
    dorm_records = dorm_records.drop_duplicates()
    
    dorm_records.columns = ['id', 'time', 'direction']
    dorm_records['date'] = dorm_records['time'].map(lambda s: get_date_info(s)[0])
    dorm_records['hour'] = dorm_records['time'].map(lambda s: get_date_info(s)[3])
    dorm_records['weekday'] = dorm_records['time'].map(lambda s: get_date_info(s)[4])
     
    
    dorm = pd.DataFrame(dorm_records.groupby(['id', 'direction'])['date'].count())
    dorm['unique_date'] = dorm_records.groupby(['id', 'direction'])['date'].apply(lambda x: len(x.unique()))
    dorm['date_avg'] = dorm.date / dorm.unique_date
    dorm['first_hour'] = dorm_records.groupby(['id', 'direction'])['hour'].min()
    dorm['last_hour'] = dorm_records.groupby(['id', 'direction'])['hour'].max()
    dorm['weekend'] = dorm_records[(dorm_records.weekday>=4) & (dorm_records.weekday<=6)].groupby(['id', 'direction'])['date'].count()
    dorm['unique_weekend'] = dorm_records[(dorm_records.weekday>=4) & (dorm_records.weekday<=6)].groupby(['id', 'direction'])['date'].apply(lambda x: len(x.unique()))
    dorm['weekend_avg'] = dorm.weekend / dorm.unique_weekend
    dorm['weekend_first_hour'] = dorm_records[(dorm_records.weekday>=4) & (dorm_records.weekday<=6)].groupby(['id', 'direction'])['hour'].min()
    dorm['weekend_last_hour'] = dorm_records[(dorm_records.weekday>=4) & (dorm_records.weekday<=6)].groupby(['id', 'direction'])['hour'].max()
    dorm = dorm.reset_index()
    
    dorm_feature = pd.DataFrame(dorm.id.unique(), columns = ['id'])
    print "  -> out dorm"
    out = dorm[['id', 'date', 'date_avg', 'first_hour', 'last_hour', \
                'weekend', 'weekend_avg', 'weekend_first_hour', 'weekend_last_hour']][dorm.direction==1]
    dorm_feature = pd.merge(dorm_feature, out, how='left', on='id')
    print "  -> in dorm"
    inside = dorm[['id', 'date', 'date_avg', 'first_hour', 'last_hour', \
                'weekend', 'weekend_avg', 'weekend_first_hour', 'weekend_last_hour']][dorm.direction==0]
    dorm_feature = pd.merge(dorm_feature, inside, how='left', on='id')
    dorm_feature.columns = ['id', 'sum_1_dorm', 'avg_1_dorm', 'first_hour_1_dorm', 'last_hour_1_dorm',
                            'sum_weekedend_1_dorm', 'avg_weekedend_1_dorm', 'first_hour_weekedend_1_dorm', 'last_hour_weekedend_1_dorm',
                            'sum_0_dorm', 'avg_0_dorm', 'first_hour_0_dorm', 'last_hour_0_dorm',
                            'sum_weekedend_0_dorm', 'avg_weekedend_0_dorm', 'first_hour_weekedend_0_dorm', 'last_hour_weekedend_0_dorm']

    dorm_feature.to_csv(join(feature_path, 'dorm_features_'+ flag +'.csv'), index=False)
    

def extract_ranking_lda_dynamic(train_feature, test_feature):
    train_id = train_feature.id
    test_id = test_feature.id
    y = train_feature['money']
    train = feature_selection(train_feature, ['_RANKING'])
    test = feature_selection(test_feature, ['_RANKING'])
    
    lda = LinearDiscriminantAnalysis()
    lda.fit(train, y)
    train = pd.DataFrame({'id' : train_id, 
                          'RANKING_lda_0' : lda.transform(train)[:, 0],
                          'RANKING_lda_1' : lda.transform(train)[:, 1],
                          'RANKING_lda_2' : lda.transform(train)[:, 2]})
    test = pd.DataFrame({'id' : test_id, 
                          'RANKING_lda_0' : lda.transform(test)[:, 0],
                          'RANKING_lda_1' : lda.transform(test)[:, 1],
                          'RANKING_lda_2' : lda.transform(test)[:, 2]})

    train_feature = pd.merge(train_feature, train, how='left', on='id')
    test_feature = pd.merge(test_feature, test, how='left', on='id')
    
    return train_feature, test_feature
    

def _rank(feature):
    distinct_val = set(feature)
    sorted_distinct_val = sorted(distinct_val, key=lambda x: x)
    ranking_feature = [list(sorted_distinct_val).index(x) + 1 for x in feature.tolist()]
    
    return ranking_feature


def transform_ranking_feature():
    all_feature = get_all_continus_feature()
    #fill na
    all_feature = all_feature.fillna(0)
    
    all_ranking_feature = all_feature[['id']]
    for _feat in all_feature.columns:
        if _feat == 'id':
            continue
        rank_feat = _rank(all_feature[_feat])
        new_feat = _feat.upper() + '_RANKING'
        all_ranking_feature[new_feat] = rank_feat
    
    # print all_ranking_feature.head()
    all_ranking_feature.to_csv(join(feature_path, 'RANKING_features.csv'), index=False)
    
    
def extract_id_feature():
    id_train = pd.read_csv(join(train_path, 'subsidy_train.txt'), sep=',', header=None)
    id_train.columns = ['id', 'money']
    id_train = id_train[['id']]
    id_test = pd.read_csv(join(test_path, 'studentID_test.txt'), sep=',', header=None)
    id_test.columns = ['id']
    
    id_train_test = pd.concat([id_train, id_test])
    id_train_test = id_train_test.drop_duplicates()
    
    nbin_scale = [3, 6, 12, 24, 36, 72, 144, 288, 512, 1024, 10000]
    
    for i, scale in enumerate(nbin_scale):
        cat_i = pd.cut(id_train_test[['id']], scale)
        feat_name = 'id_' + str(i)
        id_train_test[feat_name] = cat_i.codes
        
    id_train_test.to_csv(join(feature_path, 'id_features.csv'), index=False)
    
    
def extract_score_feature():
    score_train = pd.read_csv(join(train_path, 'score_train.txt'), sep=',', header=None)
    score_train.columns = ['id', 'college', 'score']
    score_test = pd.read_csv(join(test_path, 'score_test.txt'), sep=',', header=None)
    score_test.columns = ['id', 'college', 'score']
    score_train_test = pd.concat([score_train, score_test])
    score_train_test = score_train_test.drop_duplicates()
    
    
    college = pd.DataFrame(score_train_test.groupby(['college'])['score'].max())
    college = college.reset_index()
    college.columns = ['college', 'num']
    
    score_train_test = pd.merge(score_train_test, college, how='left', on='college')
    score_train_test['order'] = score_train_test['score'] / score_train_test['num']
    score_train_test.columns = ['id', 'academy', 'score_rank', 'nStudent_of_academy', 'rate_rank']
    
    score_train_test.to_csv(join(feature_path, 'score_features.csv'), index=False)


def extract_card_feature(flag):
    card_file = {'train' : join(train_path, 'card_train.txt'),
             'test' : join(test_path, 'card_test.txt')}
    
    card_records = pd.read_csv(card_file[flag], sep=',', header=None)
    card_records.columns = ['id', 'consume', 'where', 'how', 'time', 'amount', 'remainder']
    # pre process
    card_records = card_records.drop_duplicates()
    # card_records = card_records.dropna()
    card_records['amount'] = card_records['amount'].abs()
    card_records['remainder'] = card_records['remainder'].abs()
    card_records['diff'] = card_records['remainder'] - card_records['amount']
    
    # pos type feats
    print '  -> pos type feats'
    card = pd.DataFrame(card_records.groupby(['id', 'consume', 'how'])['amount'].max())
    card.columns = ['amount_max']
    card['amount_min'] = card_records.groupby(['id', 'consume', 'how'])['amount'].min()
    card['amount_avg'] = card_records.groupby(['id', 'consume', 'how'])['amount'].mean()
    card['amount_sum'] = card_records.groupby(['id', 'consume', 'how'])['amount'].sum()
    card['amount_num'] = card_records.groupby(['id', 'consume', 'how'])['amount'].count()
    card = card.reset_index()
    how_list = ["食堂", "超市", "开水", "图书馆", "洗衣房", "文印中心", "淋浴", "教务处", "校车", "校医院", "其他"]
    card = card[card['how'].isin(how_list)]
    card_feature = pd.DataFrame(card['id'].unique(),columns=['id'])
    for _type in how_list:
        pos_type = card[['id', 'amount_max', 'amount_min', 'amount_num', 'amount_sum', 'amount_avg']][(card.consume=='POS消费')&(card.how==_type)]
        pos_type.columns = ['id', 'max_'+_type+'_pos',\
                            'min_'+_type+'_pos', 'num_'+_type+'_pos',\
                            'sum_'+_type+'_pos', 'avg_'+_type+'_pos']
        card_feature = pd.merge(card_feature, pos_type, how='left', on='id')
    del pos_type
        
    card = pd.DataFrame(card_records.groupby(['id', 'consume'])['amount'].max())
    card.columns = ['amount_max']
    card['amount_min'] = card_records.groupby(['id', 'consume'])['amount'].min()
    card['amount_avg'] = card_records.groupby(['id', 'consume'])['amount'].mean()
    card['amount_sum'] = card_records.groupby(['id', 'consume'])['amount'].sum()
    card['amount_num'] = card_records.groupby(['id', 'consume'])['amount'].count()
    card = card.reset_index()
    
    # pos all feats
    print '  -> pos all feats'
    pos_all = card[['id', 'amount_max', 'amount_min', 'amount_num', 'amount_sum', 'amount_avg']][card.consume=='POS消费']
    pos_all.columns = ['id', 'max_all_pos', 'min_all_pos', 'num_all_pos', 'sum_all_pos', 'avg_all_pos']
    card_feature = pd.merge(card_feature, pos_all, how='left', on='id')
    del pos_all
    
    # transfer feats
    print '  -> transfer feats'
    transfer = card[['id', 'amount_max', 'amount_min', 'amount_avg', 'amount_sum', 'amount_num']][card.consume=='圈存转账']
    transfer.columns = ['id', 'max_transfer', 'min_transfer', 'avg_transfer', 'sum_transfer', 'num_transfer']
    card_feature = pd.merge(card_feature, transfer, how='left', on='id')
    del transfer
    
    # payget feats
    print '  -> payget feats'
    payget = card[['id', 'amount_max', 'amount_min', 'amount_avg', 'amount_sum', 'amount_num']][card.consume=='支付领取']
    payget.columns = ['id', 'max_payget', 'min_payget', 'avg_payget', 'sum_payget', 'num_payget']
    card_feature = pd.merge(card_feature, payget, how='left', on='id')
    del payget
    
    # card open
    print '  -> card open feats'
    card_open = card[['id', 'amount_num']][card.consume=='卡片开户']
    card_open.columns = ['id', 'num_of_open_card']
    card_feature = pd.merge(card_feature, card_open, how='left', on='id')
    del card_open
    
    # card close
    print '  -> card close feats'
    card_close = card[['id', 'amount_num']][card.consume=='卡片销户']
    card_close.columns = ['id', 'num_of_close_card']
    card_feature = pd.merge(card_feature, card_close, how='left', on='id')
    del card_close
    
    # card lost
    print '  -> card lost feats'
    card_lost = card[['id', 'amount_num']][card.consume=='卡挂失']
    card_lost.columns = ['id', 'num_of_lost_card']
    card_feature = pd.merge(card_feature, card_lost, how='left', on='id')
    del card_lost
    
    # recharge
    print '  -> recharge feats'
    recharge_amount = card[['id', 'amount_max', 'amount_min', 'amount_avg', 'amount_sum', 'amount_num']][card.consume=='卡充值']
    recharge_amount.columns = ['id', "max_account_recharge", "min_account_recharge", "avg_account_recharge", "sum_account_recharge", "num_account_recharge"]
    card_feature = pd.merge(card_feature, recharge_amount, how='left', on='id')
    card = pd.DataFrame(card_records.groupby(['id', 'consume'])['diff'].max())
    card.columns = ['diff_max']
    card['diff_min'] = card_records.groupby(['id', 'consume'])['diff'].min()
    card['diff_avg'] = card_records.groupby(['id', 'consume'])['diff'].mean()
    card['diff_sum'] = card_records.groupby(['id', 'consume'])['diff'].sum()
    card = card.reset_index()
    recharge_diff = card[['id', 'diff_max', 'diff_min', 'diff_avg', 'diff_sum']][card.consume=='卡充值']
    recharge_diff.columns = ['id', "max_diff_recharge", "min_diff_recharge", "avg_diff_recharge", "sum_diff_recharge"]
    card_feature = pd.merge(card_feature, recharge_diff, how='left', on='id')
    del recharge_amount
    del recharge_diff
    
    # balance
    print '  -> balance feats'
    card = pd.DataFrame(card_records.groupby(['id'])['remainder'].max())
    card['remainder_min'] = card_records[card_records.remainder > 0].groupby(['id'])['remainder'].min()
    card['remainder_avg'] = card_records[card_records.remainder > 0].groupby(['id'])['remainder'].mean()
    card.columns = ['max_balance', 'min_balance', 'avg_balance']
    card = card.reset_index()
    card_feature = pd.merge(card_feature, card, how='left', on='id')
    del card
    
    print '  -> save'
    # card_feature = card_feature[['id', 'max_食堂_pos', 'min_食堂_pos', 'num_食堂_pos', 'sum_食堂_pos', 'avg_食堂_pos',\
    #                            'max_超市_pos', 'min_超市_pos', 'num_超市_pos', 'sum_超市_pos', 'avg_超市_pos',\
    #                            'max_开水_pos', 'min_开水_pos', 'num_开水_pos', 'sum_开水_pos', 'avg_开水_pos',\
    #                            'max_图书馆_pos', 'min_图书馆_pos', 'num_图书馆_pos', 'sum_图书馆_pos', 'avg_图书馆_pos',\
    #                            'max_洗衣房_pos', 'min_洗衣房_pos', 'num_洗衣房_pos', 'sum_洗衣房_pos', 'avg_洗衣房_pos',\
    #                            'max_文印中心_pos', 'min_文印中心_pos', 'num_文印中心_pos', 'sum_文印中心_pos', 'avg_文印中心_pos',\
    #                            'max_淋浴_pos', 'min_淋浴_pos', 'num_淋浴_pos', 'sum_淋浴_pos', 'avg_淋浴_pos',\
    #                            'max_教务处_pos', 'min_教务处_pos', 'num_教务处_pos', 'sum_教务处_pos', 'avg_教务处_pos',\
    #                            'max_校车_pos', 'min_校车_pos', 'num_校车_pos', 'sum_校车_pos', 'avg_校车_pos',\
    #                            'max_校医院_pos', 'min_校医院_pos', 'num_校医院_pos', 'sum_校医院_pos', 'avg_校医院_pos',\
    #                            'max_其他_pos', 'min_其他_pos', 'num_其他_pos', 'sum_其他_pos', 'avg_其他_pos',\
    #                            'max_all_pos', 'min_all_pos', 'num_all_pos', 'sum_all_pos', 'avg_all_pos',\
    #                            'max_balance', 'min_balance', 'avg_balance',\
    #                            'max_payget', 'min_payget', 'num_payget', 'sum_payget', 'avg_payget',\
    #                            'num_of_lost_card', 'max_account_recharge', 'min_account_recharge',\
    #                            'num_account_recharge', 'sum_account_recharge', 'avg_account_recharge',\
    #                            'max_diff_recharge', 'min_diff_recharge', 'sum_diff_recharge', 'avg_diff_recharge',\
    #                            'max_transfer', 'min_transfer', 'num_transfer', 'sum_transfer','avg_transfer',\
    #                            'num_of_open_card', 'num_of_close_card']]
    card_feature.to_csv(join(feature_path, 'card_features_'+ flag +'.csv'), index=False)
