#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 18:17:43 2018

@author: travisbarton
"""

# First attempt at a recomender system
import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances



header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep = '\t', names = header)

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]


train_data, test_data = cv.train_test_split(df, test_size = .25)



''' this is a clever for loop that creates a data base the size of the original
data, but only fills in the entries that appear in the training data, leaving 
the test data entries totally blank '''
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]



# the two different ways of thinking about stuff. 
user_similarity = pairwise_distances(train_data_matrix, metric = 'cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric = 'cosine')

