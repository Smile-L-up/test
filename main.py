

import argparse
import random
import numpy as np
import os
# from engines.train import train
from engines.fabic_train import fAbic_train
# from engines.FBIC_train import fbic_train
from engines.data import DataManager
from engines.configure import Configure
from engines.utils.logger import get_logger
import tensorflow as tf


def set_env(configures):
    random.seed(configures.seed)
    np.random.seed(configures.seed)

    # os.environ['CUDA_VISIBLE_DEVICES'] = configures.CUDA_VISIBLE_DEVICES


def fold_check(configures):
    datasets_fold = 'datasets_fold'
    assert hasattr(configures, datasets_fold), 'item datasets_fold not configured'

    if not os.path.exists(configures.datasets_fold):
        print('datasets fold not found')
        exit(1)

    checkpoints_dir = 'checkpoints_dir'
    if not os.path.exists(configures.checkpoints_dir) or not hasattr(configures, checkpoints_dir):
        print('checkpoints fold not found, creating...')
        paths = configures.checkpoints_dir.split('/')
        if len(paths) == 2 and os.path.exists(paths[0]) and not os.path.exists(configures.checkpoints_dir):
            os.mkdir(configures.checkpoints_dir)
        else:
            os.mkdir('checkpoints')

    vocabs_dir = 'vocabs_dir'
    if not os.path.exists(configures.vocabs_dir):
        print('vocabs fold not found, creating...')
        if hasattr(configures, vocabs_dir):
            os.mkdir(configures.vocabs_dir)
        else:
            os.mkdir(configures.datasets_fold + '/vocabs')

    log_dir = 'log_dir'
    if not os.path.exists(configures.log_dir):
        print('log fold not found, creating...')
        if hasattr(configures, log_dir):
            os.mkdir(configures.log_dir)
        else:
            os.mkdir(configures.datasets_fold + '/vocabs')

def Set_GPU_Memory_Growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 设置 GPU 显存占用为按需分配
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # 异常处理
            print(e)
    else:
        print('No GPU')
		
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     growth = True
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=10240)])
#             # tf.config.experimental.set_memory_growth(gpu, growth)
#
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)
        # raise e

if __name__ == '__main__':
    Set_GPU_Memory_Growth()
    parser = argparse.ArgumentParser(description='Tuning with BiLSTM+CRF')
    parser.add_argument('--config_file', default='system.config', help='Configuration File')
    args = parser.parse_args()
    configs = Configure(config_file=args.config_file)

    fold_check(configs)
    logger = get_logger(configs.log_dir)
    configs.show_data_summary(logger)
    set_env(configs)
    dataManager = DataManager(configs, logger)
    # train(configs, dataManager, logger)
    # fbic_train(configs, dataManager, logger)
    fAbic_train(configs, dataManager, logger)
