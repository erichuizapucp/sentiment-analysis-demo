import os
import tensorflow as tf
import numpy as np

from argparse import ArgumentParser

from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
from tensorflow.keras.optimizers import SGD

from utils.common_utils import get_dirs, VOCABULARY_LENGTH, EMBEDDING_DIM, DEFAULT_BATCH_SIZE, \
    DEFAULT_ENCODED_TF_RECORD_FILE_NAME, DEFAULT_NO_EPOCHS, DEFAULT_TAKE_SIZE, DEFAULT_SHUFFLE_BUFFER_SIZE
from utils.tf_record_utils import deserialize_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PRE_TRAINED_MODELS_DIR = 'pre_trained_models'
WORD_EMBEDDINGS_MODEL_DIR_NAME = 'sentiment_analysis'
WORD_EMBEDDINGS_TRAINING_HISTORY_FILE_NAME = 'sentiment_analysis.npy'


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
    dataset = deserialize_dataset(data_file_path, is_encoded=True)

    train_data = dataset.skip(take_size).shuffle(DEFAULT_SHUFFLE_BUFFER_SIZE)
    train_data = train_data.padded_batch(batch_size, padded_shapes=([-1], []))

    test_data = dataset.take(take_size)
    test_data = test_data.padded_batch(batch_size, padded_shapes=([-1], []))

    return train_data, test_data


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
    train_dataset, val_dataset = get_dataset()

    # get the model definition (architecture)
    model = get_model()

    # compile the model adding an optimized, a loss function and metrics (e.g. accuracy)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # start the model training for a given number of epochs (e.g. 50)
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset, validation_steps=20)

    if os.path.exists(saved_model_history_path):
        os.remove(saved_model_history_path)

    # save model training history for later usage
    np.save(saved_model_history_path, history.history)

    # save model (weights, variables and computation graph) for later usage
    model.save(saved_model_path)


if __name__ == '__main__':
    work_dir, train_data_dir = get_dirs()

    args = get_cmd_args()
    data_file = args.data_file
    batch_size = args.batch_size
    epochs = args.epochs
    take_size = args.take_size

    data_file_path = os.path.join(train_data_dir, data_file)
    saved_model_path = os.path.join(work_dir, PRE_TRAINED_MODELS_DIR, WORD_EMBEDDINGS_MODEL_DIR_NAME)
    saved_model_history_path = os.path.join(work_dir, PRE_TRAINED_MODELS_DIR,
                                            WORD_EMBEDDINGS_TRAINING_HISTORY_FILE_NAME)

    # train the model and save results for later usage
    train_model()
