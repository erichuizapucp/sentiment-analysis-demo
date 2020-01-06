import tensorflow as tf
import os

from nltk.corpus import stopwords
from argparse import ArgumentParser
from utils.tf_record_utils import serialize_dataset
from utils.common_utils import get_dirs, DEFAULT_TF_RECORD_FILE_NAME

DEFAULT_SHUFFLE_BUFFER_SIZE = 50000
DEFAULT_DATA_FILE_NAMES = 'merged_neg.txt,merged_pos.txt'

REPLACE_NO_SPACE_REG_EX = '[.;:*&!\'?,\"()\[\]]'  # remove punctuation chars
REPLACE_WITH_SPACE_REG_EX = '(<br\s*/><br\s*/>)|(\-)|(\/)'  # replace <br /> tags with blank spaces

english_stop_words = stopwords.words('english')


def get_cmd_args():
    parser = ArgumentParser()
    parser.add_argument('-t', '--type', help='Type', required=True)
    parser.add_argument('-df', '--data_files', help='Data files', default=DEFAULT_DATA_FILE_NAMES)
    parser.add_argument('-bf', '--shuffle_buffer_size', help='Dataset shuffle buffer size',
                        default=DEFAULT_SHUFFLE_BUFFER_SIZE)
    parser.add_argument('-of', '--output_file_name', help='Output file name', default=DEFAULT_TF_RECORD_FILE_NAME)
    return parser.parse_args()


def main():
    args = get_cmd_args()
    data_file_names = args.data_files.split(',')
    shuffle_buffer_size = int(args.shuffle_buffer_size)
    output_file_name = args.output_file_name
    arg_type = args.type

    working_dir, samples_data_dir = get_dirs(arg_type)

    labeled_comments_datasets = []
    for index, data_file_name in enumerate(data_file_names):
        data_file_path = os.path.join(samples_data_dir, data_file_name)

        # add raw samples (lines) to the dataset
        comment_line_dataset = tf.data.TextLineDataset(data_file_path)

        # pre process samples e.g. (removing invalid characters, stop words)
        comment_line_dataset = comment_line_dataset.map(tf_pre_processing,
                                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # add a label to each line (comment) where 0 is negative and 1 is positive
        comment_line_dataset = comment_line_dataset.map(lambda sample: get_labeled_sample(sample, index),
                                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

        labeled_comments_datasets.append(comment_line_dataset)

    # concatenate all the resulting datasets in a single dataset
    all_labeled_comments_dataset = labeled_comments_datasets[0]
    for labeled_data_set in labeled_comments_datasets[1:]:
        all_labeled_comments_dataset = all_labeled_comments_dataset.concatenate(labeled_data_set)
    all_labeled_comments_dataset = all_labeled_comments_dataset.shuffle(shuffle_buffer_size,
                                                                        reshuffle_each_iteration=False)

    output_file_path = os.path.join(samples_data_dir, output_file_name)
    serialize_dataset(all_labeled_comments_dataset, output_file_path)


def pre_processing(sample):
    # lower string characters
    sample = tf.strings.lower(sample)

    # remove invalid characters (e.g. HTML tags)
    sample = tf.strings.regex_replace(sample, REPLACE_NO_SPACE_REG_EX, '')
    sample = tf.strings.regex_replace(sample, REPLACE_WITH_SPACE_REG_EX, ' ')

    # remove stop words
    sample = tf.strings.regex_replace(sample, r'\b(' + r'|'.join(english_stop_words) + r')\b\s*', '')
    sample = tf.strings.regex_replace(sample, ' +', ' ')

    return sample


def tf_pre_processing(sample):
    return tf.py_function(pre_processing, [sample], tf.string)


def get_labeled_sample(sample, index):
    return sample, index


if __name__ == '__main__':
    main()
