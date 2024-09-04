import os 
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker


current_dir = os.path.dirname(os.path.realpath(__file__))
BASE_FOLDER = os.path.abspath(os.path.join(current_dir, '../../'))


class MovielensConfig:
    
    """ 
    config dataset
    """ 
    
    DATA_SOURCE = os.path.join(BASE_FOLDER, 'storage', "dataset1M")
    DATA = os.path.join(BASE_FOLDER, 'storage')
    DATA_STREAM = os.path.join(BASE_FOLDER, 'storage', 'data_flow', 'stream_eval_users.csv')
    DATA_TRAIN = os.path.join(BASE_FOLDER, 'storage', 'data_flow', 'train_users_dict.csv')
    
    """ 
    config weights 
    """ 
    USER_MOVE_WEIGHTS = '/Users/minhtuan/Documents/Documents/Work/Hanoi/crawler/X/movielens/airflow_folder/drl_melody/storage/save_weights/u_m_model.weights.h5' #os.path.join(BASE_FOLDER, "storage", "save_weights", "u_m_model.weights.h5")
    SAVE_MODEL_PATH = os.path.join(BASE_FOLDER, "storage", "save_models")

    SAVE_ACTOR = (os.path.join(SAVE_MODEL_PATH,'actor_10_fixed.weights.h5'))
    SAVE_CRITIC = (os.path.join(SAVE_MODEL_PATH, 'critic_10_fixed.weights.h5'))

    """ 
    config Redis path
    """
    RF_NEXT_VIDEO_PATH_PERMANENCE = os.path.join(BASE_FOLDER, 'storage', 'next_video_df.txt')
    RF_NEXT_VIDEO_PATH = os.path.join(BASE_FOLDER, 'storage', "mc_nextvideo.txt")
    RF_ITEMS_IDS = os.path.join(BASE_FOLDER, 'storage', 'items_ids.txt')

    """ 
    config params
    """ 
    STATE_SIZE = 10
    EMBEDDING_SIZE = 100
    EPSILON_FOR_PRIRITY = 1e-6
    BATCH_SIZE = 32
    MAX_EPISODE_NUM = 10
    LOG_PERIOD = 180
    EMBEDDING_DIM = 100
    MAX_EPOCH = 1
    MOVIE_BATCH_SIZE = 128

    BATCH_TIME = 10
    QUEUE_SIZE = 30000

    INIT_USER_BATCH_SIZE = 64
    FINAL_USER_BATCH_SIZE = 1024

    USERS_NUM = 6041
    ITEMS_NUM = 3953