'''
Created on Dec 25, 2016

@author: loapui
'''

import time
from os.path import join

from modelSelection.cascaded import cascadedClassifier
from featureEngineer import get_all_feature
from featureEngineer.base import result_path
from postprocess import fix_result_via_rule


if __name__ == '__main__':
    
    train_feature = get_all_feature('train')
    test_feature =  get_all_feature('test')
    
    # fill na
    train_feature = train_feature.fillna(0)
    test_feature = test_feature.fillna(0)
    
    studentID_label = cascadedClassifier(train_feature, test_feature)
    
    file_name = time.strftime("%Y-%m-%d", time.localtime(time.time())) + ".csv"
    studentID_label.to_csv(join(result_path, file_name), index=False)
    
    fix_result_via_rule()
    