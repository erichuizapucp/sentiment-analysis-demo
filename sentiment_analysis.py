import os
import tensorflow as tf
import numpy as np

from argparse import ArgumentParser

from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional

from utils.common_utils import get_dirs, VOCABULARY_LENGTH, EMBEDDING_DIM, DEFAULT_BATCH_SIZE, \
    DEFAULT_ENCODED_TF_RECORD_FILE_NAME, DEFAULT_NO_EPOCHS, DEFAULT_TAKE_SIZE, DEFAULT_SHUFFLE_BUFFER_SIZE
from utils.tf_record_utils import deserialize_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PRE_TRAINED_MODELS_DIR = 'pre_trained_models'
SAVED_MODEL_NAME = 'sentiment_analysis.h5'
TRAINING_HISTORY_FILE_NAME = 'training_history.npy'
# CHECKPOINT_DIR = 'training_checkpoints'


def get_cmd_args():
    parser = ArgumentParser()
    parser.add_argument('-df', '--data_file', help='Data files', default=DEFAULT_ENCODED_TF_RECORD_FILE_NAME)
    parser.add_argument('-bs', '--batch_size', help='Batch size', default=DEFAULT_BATCH_SIZE)
    parser.add_argument('-of', '--epochs', help='Epochs', default=DEFAULT_NO_EPOCHS)
    parser.add_argument('-bf', '--shuffle_buffer_size', help='Dataset shuffle buffer size',
                        default=DEFAULT_SHUFFLE_BUFFER_SIZE)
    parser.add_argument('-ts', '--take_size', help='Dataset split take size', default=DEFAULT_TAKE_SIZE)

    return parser.parse_args()


def get_dataset():
    train_dataset = deserialize_dataset(train_data_file_path, is_encoded=True)
    test_dataset = deserialize_dataset(test_data_file_path, is_encoded=True)

    # train_data = dataset.skip(take_size).shuffle(DEFAULT_SHUFFLE_BUFFER_SIZE)
    train_dataset = train_dataset.padded_batch(batch_size, padded_shapes=([-1], []))
    # train_dataset = train_dataset.padded_batch(batch_size, padded_shapes=train_dataset.output_shapes)

    # test_data = dataset.take(take_size)
    test_data = test_dataset.padded_batch(batch_size, padded_shapes=([-1], []))
    # test_data = test_dataset.padded_batch(batch_size, padded_shapes=test_dataset.output_shapes)

    return train_dataset, test_data


def get_model():
    inputs = Input(shape=(DEFAULT_BATCH_SIZE,))
    x = Embedding(VOCABULARY_LENGTH, EMBEDDING_DIM, input_length=DEFAULT_BATCH_SIZE)(inputs)
    x = Bidirectional(LSTM(128))(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    return Model(inputs=inputs, outputs=output)


def train_model():
    # split the train dataset into train and validation (only 5000 comments will be used for validation the rest
    # will be used for training)
    train_dataset, test_dataset = get_dataset()

    # get the model definition (architecture)
    model = get_model()

    # compile the model adding an optimized, a loss function and metrics (e.g. accuracy)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # Checkpoint CallBack
    # checkpoint_prefix = os.path.join(checkpoint_dir_path, "ckpt_{epoch}")
    # checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_prefix,
    #     save_weights_only=True)

    # start the model training for a given number of epochs (e.g. 50)
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset, validation_steps=30)  #, callbacks=[checkpoint_callback])

    if os.path.exists(saved_model_history_path):
        os.remove(saved_model_history_path)

    # save model training history for later usage
    np.save(saved_model_history_path, history.history)

    # save model (weights, variables and computation graph) for later usage
    model.save(saved_model_path)

    model.evaluate(test_dataset)


if __name__ == '__main__':
    work_dir, train_data_dir = get_dirs(arg_type='train')
    _, test_data_dir = get_dirs(arg_type='test')

    args = get_cmd_args()
    data_file = args.data_file
    batch_size = args.batch_size
    epochs = args.epochs
    take_size = args.take_size

    train_data_file_path = os.path.join(train_data_dir, data_file)
    test_data_file_path = os.path.join(test_data_dir, data_file)

    saved_model_path = os.path.join(work_dir, PRE_TRAINED_MODELS_DIR, SAVED_MODEL_NAME)
    saved_model_history_path = os.path.join(work_dir, PRE_TRAINED_MODELS_DIR,
                                            TRAINING_HISTORY_FILE_NAME)
    # checkpoint_dir_path = os.path.join(work_dir, PRE_TRAINED_MODELS_DIR, CHECKPOINT_DIR)

    # train the model and save results for later usage
    train_model()
