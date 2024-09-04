## run 

import tensorflow as tf
import numpy as np
from tensorflow.python.ops.gen_math_ops import Exp
from datetime import datetime
import os
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import time
import random
import sys 
# sys.path.append('/home/tuannm84/Desktop/longbien/Project/MC/')

from model.actor import Actor
from model.critic import Critic
from model.enviroment import StimulateEnv
from model.ddpg import DDPG 
from model.embedding import VideoGenreEmbedding, UserVideoEmbedding
from model.ou_noise import OUNoise


Path = '/home/tuannm84/Desktop/longbien/Project/MC'

PATH_USER_DICT = os.path.join(Path, "dataset/user_dict.npy")
PATH_TRAIN_DATASET = os.path.join(Path, "dataset/train_dict.npy")
PATH_EVAL_DATSET = os.path.join(Path, "dataset/eval_dict.npy")
PATH_USER_HISTORY_LENS = os.path.join(Path, 'dataset/users_history_len_local.npy')
PATH_DICTIONARY = os.path.join(Path, "dataset/dictionary.npy")
PATH_DATA_NUMBER = os.path.join(Path, "dataset/data_number.npy")

users_dict = np.load(PATH_USER_DICT,allow_pickle='TRUE').item()
eval_users_dict = np.load(PATH_EVAL_DATSET,allow_pickle='TRUE').item()
train_users_dict = np.load(PATH_TRAIN_DATASET,allow_pickle='TRUE').item()
dictionary = np.load(PATH_DICTIONARY,allow_pickle='TRUE').item()
data_number = np.load(PATH_DATA_NUMBER,allow_pickle='TRUE').item()
users_history_lens = np.load(PATH_USER_HISTORY_LENS, allow_pickle='TRUE')
all_items = {data[0] for i, k in users_dict.items() for data in k}  ## list video toan tap data 

user_dataset = eval_users_dict
user_id = 10 
users_history_lens = len(user_dataset[user_id])
newest_watched_video = np.random.choice([i[0] for i in user_dataset[11]])
watched_videos = [data[0] for data in users_dict[user_id]]
items_ids = np.array(list(set(all_items) - set(watched_videos)))
len_items_ids = len(items_ids)
STATE_SIZE = len_items_ids ## 1445 là số lượng videos sau khi trừ đi các video đã xem trong history
num_actions = len_items_ids
output_dim = len_items_ids
users_num = data_number['users_num']
items_num = data_number['items_num']

env_prod = StimulateEnv(user_id, newest_watched_video, users_dict, users_history_lens, STATE_SIZE)
recommender  = DDPG(env_prod, users_num, items_num, num_actions, STATE_SIZE, output_dim)  # output_dim là output của State_emebedding, để 1445 vì đầu vào của actor.evaluate_actor là (1445,400)

EMBEDDING_SIZE = 100
epsilon_for_priority = 1e-6
batch_size = 32
num_actions = len_items_ids ## Number of list video to be choosed
num_actions = len_items_ids
exploration_noise = OUNoise(num_actions)


#######################################################################################
"""
CASE 1: STATE from ITEMS_IDS => feed into EVALUATE_ACTOR => ACTION  
"""
x = items_ids
array_x = np.reshape(x,[1, num_actions])
state_value = tf.convert_to_tensor(x, dtype=tf.float32)
state_value = tf.expand_dims(state_value, axis=0)



""" 
CASE 2: STATE from ITEM_IDS + STATE_REPRESENTATION + USER_EMDS => STATE => feed into EVALUATE_ACTOR => ACTION 
"""
user_eb = recommender.embedding_network.get_layer('user_embedding')(np.array(user_id))
items_eb = recommender.embedding_network.get_layer('video_embedding')(np.array(items_ids))
state = recommender.srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(items_eb, axis=0)])
action = recommender.evaluate_actor(state)



#######################################################################################

action = recommender.evaluate_actor(state_value)
noise = exploration_noise.noise()
action = action[0] + noise
recommended_item = recommender.recommend_item(action, all_items, env_prod.old_watched, top_k= 5)
next_items_ids_embs, reward, done, _ = env_prod.step(recommended_item)
reward = np.sum(reward)

# agent.add_experience(state_value, next_items_ids,action,reward,done)



""" 
2 cách trên đang ra cùng 1 kết quả (xem tại jupyter notebook)

tiếp theo: Hoàn thiện Circle workflow - phần train và phần minibatch thử chạy xem tiếp ntn và cái replay hoạt động ổn k ra sao

"""