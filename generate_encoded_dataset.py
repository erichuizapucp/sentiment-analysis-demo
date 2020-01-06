import os

from utils.common_utils \
    import get_dirs, DEFAULT_TF_RECORD_FILE_NAME, DEFAULT_BATCH_SIZE, DEFAULT_ENCODED_TF_RECORD_FILE_NAME, \
    DEFAULT_VOCABULARY_FILE
from argparse import ArgumentParser
from utils.tf_record_utils import deserialize_dataset, serialize_dataset
from utils.encoding_utils import encode_dataset

DATA_DIR = 'data'


def get_cmd_args():
    parser = ArgumentParser()
    parser.add_argument('-t', '--type', help='Type', required=True)
    parser.add_argument('-df', '--data_file', help='Data files', default=DEFAULT_TF_RECORD_FILE_NAME)
    parser.add_argument('-bs', '--batch_size', help='Batch size', default=DEFAULT_BATCH_SIZE)
    parser.add_argument('-of', '--output_file', help='Encoded Data File',
                        default=DEFAULT_ENCODED_TF_RECORD_FILE_NAME)
    parser.add_argument('-vf', '--vocabulary_file', help='Vocabulary File', default=DEFAULT_VOCABULARY_FILE)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_cmd_args()
    data_file_name = args.data_file
    batch_size = args.batch_size
    output_file_name = args.output_file
    vocabulary_file_name = args.vocabulary_file
    arg_type = args.type

    working_dir, samples_data_dir = get_dirs(arg_type)

    data_file_path = os.path.join(samples_data_dir, data_file_name)

    # deserialize dataset (this contains decoded data)
    dataset = deserialize_dataset(data_file_path)
    vocabulary_file_path = os.path.join(working_dir, DATA_DIR, vocabulary_file_name)
    encoded_dataset = encode_dataset(dataset, vocabulary_file_path)
    # sample_text, sample_labels = next(iter(encoded_dataset))

    output_file_path = os.path.join(samples_data_dir, output_file_name)
    serialize_dataset(encoded_dataset, output_file_path, is_encoded=True)
