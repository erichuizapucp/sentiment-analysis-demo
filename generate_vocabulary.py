import os
import tensorflow_datasets as tfds
import csv

from utils.common_utils import get_dirs, DEFAULT_TF_RECORD_FILE_NAME, DEFAULT_VOCABULARY_FILE, \
    DEFAULT_SHUFFLE_BUFFER_SIZE
from argparse import ArgumentParser
from utils.tf_record_utils import deserialize_dataset

DATA_DIR = 'data'


def get_cmd_args():
    parser = ArgumentParser()
    parser.add_argument('-df', '--data_file', help='Data files', default=DEFAULT_TF_RECORD_FILE_NAME)
    parser.add_argument('-of', '--output_file', help='Encoded Data File', default=DEFAULT_VOCABULARY_FILE)
    parser.add_argument('-bf', '--shuffle_buffer_size', help='Dataset shuffle buffer size',
                        default=DEFAULT_SHUFFLE_BUFFER_SIZE)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_cmd_args()
    data_file_name = args.data_file
    output_file_name = args.output_file
    shuffle_buffer_size = int(args.shuffle_buffer_size)

    working_dir, train_data_dir = get_dirs(arg_type='train')
    _, test_data_dir = get_dirs(arg_type='test')

    train_data_file_path = os.path.join(train_data_dir, data_file_name)
    train_dataset = deserialize_dataset(train_data_file_path)

    test_data_file_path = os.path.join(test_data_dir, data_file_name)
    test_dataset = deserialize_dataset(test_data_file_path)

    dataset = train_dataset
    dataset = dataset.concatenate(test_dataset)
    dataset = dataset.shuffle(shuffle_buffer_size, reshuffle_each_iteration=False)

    tokenizer = tfds.features.text.Tokenizer()
    vocabulary = set()  # it is a set to ensure unique words only
    for comment, _ in dataset:
        tokens = tokenizer.tokenize(comment.numpy())
        vocabulary.update(tokens)

    data_dir = os.path.join(working_dir, DATA_DIR)
    output_file_path = os.path.join(data_dir, output_file_name)
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    with open(output_file_path, mode='w') as vocabulary_csv:
        vocabulary_writer = csv.writer(vocabulary_csv)
        for entry in vocabulary:
            vocabulary_writer.writerow([entry])
