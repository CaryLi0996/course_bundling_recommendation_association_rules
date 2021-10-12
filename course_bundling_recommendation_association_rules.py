#!/usr/bin/env python
# coding: utf-8

# # Objective: recommend suitable course bundles for students with application of association rules to increase likelihood for purchase

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')

from pathlib import Path
import heapq
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

## you may need to install mlxtend
import sys
get_ipython().system('{sys.executable} -m pip install mlxtend')

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[6]:


course_df = pd.read_csv('Coursetopics.csv')
course_df['Student'] = np.arange(len(course_df))
course_df.set_index('Student', inplace=True)
course_df


# In[7]:


#create frequent itemsets
itemsets = apriori(course_df,min_support=0.02, use_colnames=True) 
#smaller support value threshold for itemsets since percent of transactions that include some necessary itemsets is low
#print(itemsets)
#converting into rules
rules = association_rules(itemsets, metric='confidence', min_threshold=0.10)
rules.sort_values(by=['lift'], ascending=False)
#lift helps see how much better  the chance of getting the consequent if you use the rule than if you select randomly.
rules = rules.drop(columns=['antecedent support', 'consequent support', 'conviction'])
rules.sort_values(by='confidence',ascending=False).head(15)


# 

# In[8]:


get_ipython().system(' pip install scikit-surprise')

from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split


# In[9]:


def convert(data):
  result = data.stack().reset_index()
  result.columns = ['userID', 'itemID', 'rating']

  return result


# In[10]:


course_df_converted = convert(course_df)
course_df_converted


# In[11]:


def get_top_n(predictions, n=10):
    # First map the predictions to each user.
    byUser = defaultdict(list)
    for p in predictions:
        byUser[p.uid].append(p)
    
    # For each user, reduce predictions to top-n
    for uid, userPredictions in byUser.items():
        byUser[uid] = heapq.nlargest(n, userPredictions, key=lambda p: p.est)
    return byUser


# In[12]:


# Convert these data set into the format required by the surprise package
# The columns must correspond to user id, item id and ratings (in that order)

reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(course_df_converted[['userID', 'itemID', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.25, random_state=1)


# # User-based filtering

# In[13]:


# compute cosine similarity between users 
sim_options = {'name': 'cosine', 'user_based': True}
algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset)

# Then predict ratings for all pairs (u, i) that are NOT in the training set.
predictions = algo.test(testset)

top_n = get_top_n(predictions, n=4)

# Print the recommended items for each user
print()
print('Top-4 recommended items for each user')
for uid, user_ratings in list(top_n.items())[:5]:
    print('User {}'.format(uid))
    for prediction in user_ratings:
        print('  Item {0.iid} ({0.est:.2f})'.format(prediction), end='')
    print()
print()


# # For student groups who have purchased the Regression and Forecast courses, what should we recommend to them? 

# In[14]:


# need to create sample test data for prediction for a student who purchased Regression and Forecast book

sample_df = pd.DataFrame({'Intro':[0], 'DataMining': [0], 'Survey': [0], 'Cat Data': [0], 'Regression':[1], 'Forecast': [1], 'DOE':[0], 'SW': [0]})
sample_df


# In[15]:


sample_df_converted = convert(sample_df)
sample_df_converted


# In[16]:


reader = Reader(rating_scale=(0,1))
data = Dataset.load_from_df(sample_df_converted[['userID', 'itemID', 'rating']], reader)

trainset, testet = train_test_split(data, test_size=1, random_state=1)


# In[17]:


predictions = algo.test(trainset.build_testset())

