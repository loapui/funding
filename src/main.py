'''
Created on Dec 25, 2016

@author: loapui
'''

import time
from os.path import join

from modelSelection.cascaded import cascadedClassifier
from featureEngineer import get_all_feature
from featureEngineer.feature_extraction import extract_card_feature, extract_dorm_feature, extract_library_feature, extract_score_feature,\
                                            extract_id_feature, extract_zero_feature, transform_ranking_feature 
from featureEngineer.base import result_path
from postprocess import fix_result_via_rule

def extract_feature():
    extract_card_feature('train')
    extract_card_feature('test')
    
    extract_dorm_feature('train')
    extract_dorm_feature('test')
    
    extract_library_feature('train')
    extract_library_feature('test')
    
    extract_score_feature()
    
    extract_id_feature()
    
    extract_zero_feature()
    
    transform_ranking_feature()
    

if __name__ == '__main__':
    
    # step 1.
    extract_feature()
    
    # step 2.
    train_feature = get_all_feature('train')
    test_feature =  get_all_feature('test')
    
    # step 3. predict
    studentID_label = cascadedClassifier(train_feature, test_feature)
    
    # step 4. save result
    file_name = time.strftime("%Y-%m-%d", time.localtime(time.time())) + ".csv"
    studentID_label.to_csv(join(result_path, file_name), index=False)
    
    fix_result_via_rule()
    