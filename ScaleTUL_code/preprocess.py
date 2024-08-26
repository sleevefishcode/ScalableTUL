import pandas as pd
from tqdm import tqdm
from ast import literal_eval
from gensim.models import Word2Vec
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from collections import defaultdict
def split_dataset(dataset_path):

    print('----------load dataset...----------')

    #---------------read_dataset--------------#
    all_dataset = pd.read_csv(dataset_path, sep=',', header=None, names=['user', 'traj', 'time', 'category'])
    
    #-------------statistical corpus----------#
    user_list = all_dataset['user'].drop_duplicates().values.tolist()
    poi_list = set()
    category_list = set()
    
    for idx, (_, traj, _, category) in tqdm(all_dataset.iterrows(), total=len(all_dataset), ncols=100):
        poi_list.update(literal_eval(traj))
        category_list.update(literal_eval(category))
    
    poi_nums = len(poi_list)
    category_nums = len(category_list)
    user_nums = len(user_list)

    user_embedding={}
    #--------split train-test dataset--------#
    train_nums = 0
    test_nums = 0
    user_traj_train, user_traj_test = {}, {}
    for user in tqdm(user_list, ncols=100):
        user_traj_train[user], user_traj_test[user] = [], []
        one_user_data = all_dataset.loc[all_dataset.user==user,:]
        one_user_data_train = one_user_data.iloc[:int(0.8*len((one_user_data)))]
        one_user_data_test = one_user_data.iloc[int(0.8*len((one_user_data))):]

        for idx , (_, one_row) in enumerate(one_user_data_train.iterrows()):
            train_nums += 1
            _, traj, time, category = one_row
            user_traj_train[user].append((literal_eval(traj), literal_eval(time), literal_eval(category), idx))
        
        for idx, (_, one_row) in enumerate(one_user_data_test.iterrows()):
            _, traj, time, category = one_row
            test_nums+=1
            user_traj_test[user].append((literal_eval(traj), literal_eval(time), literal_eval(category), idx))


    for user in user_list:
        weekly_traj,train_nums= merge_daily_to_weekly(user_traj_train[user],train_nums)
        user_traj_train[user].extend(weekly_traj)
    for user in user_list:
        two_weekly_traj,train_nums= merge_daily_to_twoweekly(user_traj_train[user],train_nums)
        user_traj_train[user].extend(two_weekly_traj)    
 
    print('-------Finish loading data!--------')
    print(train_nums, poi_nums, category_nums , user_nums)
    return user_traj_train, user_traj_test, train_nums, poi_nums, category_nums , user_nums,user_embedding

def merge_daily_to_weekly(user_traj,train_nums):
    weekly_traj = []
    week_traj, week_time, week_category = [], [], []
    week_idx = 0

    for day_traj, day_time, day_category, _ in user_traj:
        week_traj.extend(day_traj)
        week_time.extend(day_time)
        week_category.extend(day_category)
        

        if len(week_traj) >= 7:
            weekly_traj.append((week_traj, week_time, week_category, week_idx))
            train_nums += 1
            week_traj, week_time, week_category = [], [], []
            week_idx += 1

    if week_traj:
        weekly_traj.append((week_traj, week_time, week_category, week_idx))
        train_nums += 1
    
    return weekly_traj,train_nums
def merge_daily_to_twoweekly(user_traj,train_nums):
    weekly_traj = []
    week_traj, week_time, week_category = [], [], []
    week_idx = 0

    for day_traj, day_time, day_category, _ in user_traj:
        week_traj.extend(day_traj)
        week_time.extend(day_time)
        week_category.extend(day_category)
        
      
        if len(week_traj) >= 14:
            weekly_traj.append((week_traj, week_time, week_category, week_idx))
            train_nums += 1
            week_traj, week_time, week_category = [], [], []
            week_idx += 1

    if  len(week_traj)>7:
        weekly_traj.append((week_traj, week_time, week_category, week_idx))
        train_nums += 1
    
    return weekly_traj,train_nums
if __name__=='__main__':
    user_traj_train, user_traj_test, train_nums, poi_nums, category_nums , user_nums,user_embedding = split_dataset('./dataset/foursquare_NY_mini.csv',512)
    # print(user_traj_test)
    # print(train_nums, poi_nums, category_nums , user_nums)
