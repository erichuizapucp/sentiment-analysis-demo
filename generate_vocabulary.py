import os
import tensorflow_datasets as tfds
import csv

from utils.common_utils import get_dirs, DEFAULT_TF_RECORD_FILE_NAME, DEFAULT_VOCABULARY_FILE
from argparse import ArgumentParser
from utils.tf_record_utils import deserialize_dataset


def get_cmd_args():
    parser = ArgumentParser()
    parser.add_argument('-df', '--data_file', help='Data files', default=DEFAULT_TF_RECORD_FILE_NAME)
    parser.add_argument('-of', '--output_file', help='Encoded Data File',
                        default=DEFAULT_VOCABULARY_FILE)
    return parser.parse_args()


if __name__ == '__main__':
    _, train_data_dir = get_dirs()
    args = get_cmd_args()
    data_file_name = args.data_file
    output_file_name = args.output_file

    data_file_path = os.path.join(train_data_dir, data_file_name)
    dataset = deserialize_dataset(data_file_path)

    tokenizer = tfds.features.text.Tokenizer()
    vocabulary = set()  # it is a set to ensure unique words only
    for comment, _ in dataset:
        tokens = tokenizer.tokenize(comment.numpy())
        vocabulary.update(tokens)

    output_file_path = os.path.join(train_data_dir, output_file_name)
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    with open(output_file_path, mode='w') as vocabulary_csv:
        vocabulary_writer = csv.writer(vocabulary_csv)
        for entry in vocabulary:
            vocabulary_writer.writerow([entry])
