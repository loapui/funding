#coding=utf-8
'''
Created on Dec 26, 2016

@author: loapui
'''

import time
from os.path import join

import pandas as pd
from featureEngineer.base import result_path
from featureEngineer.features import get_all_feature

def _analyse_fix_count(record):
    if record[1] == record[2]:
        return 0
    else:
        return 1


def _get_result():
    result_file = join(result_path, time.strftime("%Y-%m-%d", time.localtime(time.time())) + ".csv")
    studentID_label = pd.read_csv(result_file)
    
    return studentID_label


def rule(record, id_set): 
    fix_label = record[1]
    if record[0] in id_set:
        fix_label = 0
    
    return fix_label
    

def fix_result_via_rule():
    studentID_label = _get_result() 
    studentID_feature = get_all_feature('test')
    
    id_set1 = studentID_feature[studentID_feature['max_食堂_pos']>90]['id'].values
    id_set2 = studentID_feature[studentID_feature['avg_食堂_pos']>10]['id'].values
    id_set3 = studentID_feature[studentID_feature['id']>22100]['id'].values
    id_set = set(list(id_set1) + list(id_set2) + list(id_set3))
    
    studentID_label['fix_subsidy'] = studentID_label.apply(rule, id_set = id_set, axis = 1)
    studentID_label['fix_count'] = studentID_label.apply(_analyse_fix_count, axis = 1)
    print "fix %d records" % (studentID_label['fix_count'].sum())
    
    studentID_label = studentID_label[['studentid', 'fix_subsidy']]
    studentID_label.columns = ['studentid', 'subsidy']
    
    file_name = time.strftime("%Y-%m-%d", time.localtime(time.time())) + ".csv"
    studentID_label.to_csv(join(result_path, file_name), index=False)