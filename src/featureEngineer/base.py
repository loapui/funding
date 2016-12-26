'''
Created on Dec 25, 2016

@author: loapui
'''

from os.path import join, dirname
from datetime import datetime

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler

train_path = join(dirname(dirname(dirname(__file__))), 'train')
test_path = join(dirname(dirname(dirname(__file__))), 'test')
temp_path = join(dirname(dirname(dirname(__file__))), 'temp')
feature_path = join(dirname(dirname(dirname(__file__))), 'features')
result_path = join(dirname(dirname(dirname(__file__))), 'result')

def get_date_info(date_field):
    date_s = date_field.replace("\"", "")

    date = datetime.strptime(date_s, "%Y/%m/%d %H:%M:%S")
    month = date.month
    day = date.day
    hour = date.hour
    weekday = date.weekday()
    
    return [date_s.split(" ")[0], month, day, hour, weekday]


def sample(trainX, trainY, sample_params):
    
    sampler = {'under_sample' : RandomUnderSampler(ratio = sample_params['ratio'], random_state = 1, replacement = False),
               'over_sample' : RandomOverSampler(ratio = sample_params['ratio'], random_state = 1)}
    
    X_sample, y_sample = sampler[sample_params['type']].fit_sample(trainX, trainY)
    
    return X_sample, y_sample
