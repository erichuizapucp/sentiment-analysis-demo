import os
import pathlib

from argparse import ArgumentParser

WORK_DIR_ENV_VAR = 'WORK_DIR'
WORK_DIR_DEFAULT_VALUE = './'
DATA_DIR_NAME = 'data'
TRAIN_DIR_NAME = 'train'
TEST_DIR_NAME = 'test'
NEG_DIR_NAME = 'neg'
POS_DIR_NAME = 'pos'

NEG_MERGE_FILE_NAME = 'merged_neg.txt'
POS_MERGE_FILE_NAME = 'merged_pos.txt'


def get_cmd_args():
    parser = ArgumentParser()
    parser.add_argument('-t', '--type', help='Type', required=True)
    return parser.parse_args()


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

    args = get_cmd_args()
    arg_type = args.type

    type_dir = TRAIN_DIR_NAME if arg_type == 'train' else TEST_DIR_NAME

    train_data_neg_dir = os.path.join(working_dir, DATA_DIR_NAME, type_dir, NEG_DIR_NAME)
    train_data_pos_dir = os.path.join(working_dir, DATA_DIR_NAME, type_dir, POS_DIR_NAME)

    train_neg_file_path = os.path.join(working_dir, DATA_DIR_NAME, type_dir, NEG_MERGE_FILE_NAME)
    train_pos_file_path = os.path.join(working_dir, DATA_DIR_NAME, type_dir, POS_MERGE_FILE_NAME)

    merge_comments_to_file(train_data_neg_dir, train_neg_file_path)
    merge_comments_to_file(train_data_pos_dir, train_pos_file_path)


if __name__ == '__main__':
    main()
