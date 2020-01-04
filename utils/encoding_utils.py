import tensorflow as tf
import tensorflow_datasets as tfds
import csv

encoder = None


def get_vocabulary(vocabulary_file_path):
    vocabulary = []
    with open(vocabulary_file_path, 'r') as vocabulary_file:
        reader = csv.reader(vocabulary_file)
        for row in reader:
            vocabulary.append(row[0])
    return vocabulary


def get_encoder(vocabulary):
    return tfds.features.text.TokenTextEncoder(vocabulary)


def encode(comment_tensor, label):
    encoded_text = encoder.encode(comment_tensor.numpy())
    return encoded_text, label


def encode_map_fn(comment, label):
    encoded_comment, label = tf.py_function(encode, [comment, label], (tf.int64, tf.int64))
    return encoded_comment, label


def encode_dataset(dataset, vocabulary_file_path):
    # obtain the words vocabulary based on the dataset
    vocabulary = get_vocabulary(vocabulary_file_path)

    # obtain a decoded for the dataset vocabulary
    global encoder
    encoder = get_encoder(vocabulary)

    # encode dataset
    encoded_dataset = dataset.map(lambda comment, label: encode_map_fn(comment, label))
    return encoded_dataset
