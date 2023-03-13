#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:10:30 2023

@author: ashiq
"""

#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ast 
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
# from nltk.stem.snowball import SnowballStemmer
# from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.corpus import wordnet
# from surprise import Reader, Dataset, SVD, evaluate

import warnings; warnings.simplefilter('ignore')


!ls /home/ashiq/Desktop/siri_project/Job_ML/datasets/*.tsv

apps = pd.read_csv('/home/ashiq/Desktop/siri_project/Job_ML/datasets/apps.tsv', delimiter='\t',encoding='utf-8')
user_history = pd.read_csv('/home/ashiq/Desktop/siri_project/Job_ML/datasets/user_history.tsv', delimiter='\t',encoding='utf-8')
jobs = pd.read_csv('/home/ashiq/Desktop/siri_project/Job_ML/datasets/jobs.tsv', delimiter='\t',encoding='utf-8', error_bad_lines=False)
users = pd.read_csv('/home/ashiq/Desktop/siri_project/Job_ML/datasets/users.tsv' ,delimiter='\t',encoding='utf-8')
test_users = pd.read_csv('/home/ashiq/Desktop/siri_project/Job_ML/datasets/test_users.tsv', delimiter='\t',encoding='utf-8')



apps_training = apps.loc[apps['Split'] == 'Train']


apps_testing = apps.loc[apps['Split'] == 'Test']


user_history_training = user_history.loc[user_history['Split'] =='Train']


user_history_training = user_history.loc[user_history['Split'] =='Train']
user_history_testing = user_history.loc[user_history['Split'] =='Test']
apps_training = apps.loc[apps['Split'] == 'Train']
apps_testing = apps.loc[apps['Split'] == 'Test']
users_training = users.loc[users['Split']=='Train']
users_testing = users.loc[users['Split']=='Test']