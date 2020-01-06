import os

FEATURE_NAME = 'comment'
TARGET_FEATURE_NAME = 'label'

WORK_DIR_ENV_VAR_NAME = 'WORK_DIR'
DATA_DIR_NAME = 'data'
TRAIN_DIR_NAME = 'train'
TEST_DIR_NAME = 'test'

DEFAULT_SHUFFLE_BUFFER_SIZE = 50000
DEFAULT_BATCH_SIZE = 64
VOCABULARY_LENGTH = 135034 + 1  # we add 1 because the 0 added during padding operations
EMBEDDING_DIM = 128
DEFAULT_NO_EPOCHS = 4
DEFAULT_TAKE_SIZE = 5000


DEFAULT_DATA_FILE_NAMES = ['merged_neg.txt', 'merged_pos.txt']
DEFAULT_WORK_DIR = './'
DEFAULT_TF_RECORD_FILE_NAME = 'dataset.tfrecord'
DEFAULT_ENCODED_TF_RECORD_FILE_NAME = 'encoded_dataset.tfrecord'
DEFAULT_VOCABULARY_FILE = 'vocabulary.csv'


def get_dirs(arg_type='train'):
    type_dir = TRAIN_DIR_NAME if arg_type == 'train' else TEST_DIR_NAME

    working_dir = os.getenv(WORK_DIR_ENV_VAR_NAME, DEFAULT_WORK_DIR)
    samples_data_dir = os.path.join(working_dir, DATA_DIR_NAME, type_dir)
    return working_dir, samples_data_dir
