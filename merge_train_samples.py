import os
import pathlib

WORK_DIR_ENV_VAR = 'WORK_DIR'
WORK_DIR_DEFAULT_VALUE = './'
DATA_DIR_NAME = 'data'
TRAIN_DIR_NAME = 'train'
NEG_DIR_NAME = 'neg'
POS_DIR_NAME = 'pos'

TRAIN_NEG_MERGE_FILE_NAME = 'train_merged_neg.txt'
TRAIN_POS_MERGE_FILE_NAME = 'train_merged_pos.txt'


def merge_comments_to_file(dir_path, output_file_path):
    list_files = pathlib.Path(dir_path).glob('*.txt')

    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    with open(output_file_path, 'a') as output_file:
        for index, file in enumerate(list_files):
            with file.open('r') as f:
                output_file.write('\n' + f.read() if index > 0 else '' + f.read())


def main():
    working_dir = os.getenv(WORK_DIR_ENV_VAR, WORK_DIR_DEFAULT_VALUE)

    train_data_neg_dir = os.path.join(working_dir, DATA_DIR_NAME, TRAIN_DIR_NAME, NEG_DIR_NAME)
    train_data_pos_dir = os.path.join(working_dir, DATA_DIR_NAME, TRAIN_DIR_NAME, POS_DIR_NAME)

    train_neg_file_path = os.path.join(working_dir, DATA_DIR_NAME, TRAIN_DIR_NAME, TRAIN_NEG_MERGE_FILE_NAME)
    train_pos_file_path = os.path.join(working_dir, DATA_DIR_NAME, TRAIN_DIR_NAME, TRAIN_POS_MERGE_FILE_NAME)

    merge_comments_to_file(train_data_neg_dir, train_neg_file_path)
    merge_comments_to_file(train_data_pos_dir, train_pos_file_path)


if __name__ == '__main__':
    main()
