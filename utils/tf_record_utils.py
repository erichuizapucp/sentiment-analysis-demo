import tensorflow as tf
import os

from utils.common_utils import FEATURE_NAME, TARGET_FEATURE_NAME


def bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def encoded_int64_feature(value):
    # this one allows to serialize a tensor containing a list of integers (encoded comments)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value.numpy().reshape(-1)))


def serialize_comment(comment, label, is_encoded=False):
    feature = {
        FEATURE_NAME: encoded_int64_feature(comment) if is_encoded else bytes_feature(comment),
        TARGET_FEATURE_NAME: int64_feature(label),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_comment(comment, label, is_encoded=False):
    tf_comment = tf.py_function(
        serialize_comment,
        (comment, label, is_encoded),
        tf.string)
    return tf.reshape(tf_comment, ())


def parse_dict_sample(sample):
    feature_description = {
        FEATURE_NAME: tf.io.FixedLenFeature([], tf.string, default_value=''),
        TARGET_FEATURE_NAME: tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }

    feature = tf.io.parse_single_example(sample, feature_description)
    return feature[FEATURE_NAME], feature[TARGET_FEATURE_NAME]


def encoded_parse_dict_sample(sample):
    feature_description = {
        FEATURE_NAME: tf.io.VarLenFeature(tf.int64),
        TARGET_FEATURE_NAME: tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }

    feature = tf.io.parse_single_example(sample, feature_description)

    comment = tf.sparse.to_dense(feature[FEATURE_NAME])
    label = feature[TARGET_FEATURE_NAME]

    return comment, label


def tf_parse_dict_sample(sample, is_encoded=False):
    parse_func = encoded_parse_dict_sample if is_encoded else parse_dict_sample
    return_type = (tf.int64, tf.int64) if is_encoded else (tf.string, tf.int64)
    tf_comment, tf_label = tf.py_function(parse_func, [sample], return_type)

    return tf_comment, tf_label


def serialize_dataset(dataset: tf.data.Dataset, output_file_path, is_encoded=False):
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    # serialize dataset to a .tfrecord file for further usage
    serialized_dataset = dataset.map(lambda comment, label: tf_serialize_comment(comment, label, is_encoded))
    writer = tf.data.experimental.TFRecordWriter(output_file_path)
    writer.write(serialized_dataset)


def deserialize_dataset(tf_record_file_path, is_encoded=False):
    dataset = tf.data.TFRecordDataset(tf_record_file_path).map(lambda sample: tf_parse_dict_sample(sample, is_encoded))
    return dataset
