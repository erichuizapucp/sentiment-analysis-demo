import os

FEATURE_NAME = 'comment'
TARGET_FEATURE_NAME = 'label'

WORK_DIR_ENV_VAR_NAME = 'WORK_DIR'
DATA_DIR_NAME = 'data'
TRAIN_DIR_NAME = 'train'

DEFAULT_SHUFFLE_BUFFER_SIZE = 50000
DEFAULT_BATCH_SIZE = 64
VOCABULARY_LENGTH = 93167 + 1  # we add 1 because the 0 added during padding operations
EMBEDDING_DIM = 128
DEFAULT_NO_EPOCHS = 10
DEFAULT_TAKE_SIZE = 5000


DEFAULT_DATA_FILE_NAMES = ['train_merged_neg.txt', 'train_merged_pos.txt']
DEFAULT_WORK_DIR = '../'
DEFAULT_TF_RECORD_FILE_NAME = 'train_dataset.tfrecord'
DEFAULT_ENCODED_TF_RECORD_FILE_NAME = 'encoded_train_dataset.tfrecord'
DEFAULT_VOCABULARY_FILE = 'vocabulary.csv'


def get_dirs():
    working_dir = os.getenv(WORK_DIR_ENV_VAR_NAME, DEFAULT_WORK_DIR)
    train_data_dir = os.path.join(working_dir, DATA_DIR_NAME, TRAIN_DIR_NAME)
    return working_dir, train_data_dir
