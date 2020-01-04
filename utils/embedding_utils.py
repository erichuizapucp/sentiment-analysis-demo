import io
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer


VEC_FILE_NAME = 'vecs.tsv'
VEC_META_FILE_NAME = 'meta.tsv'

PRE_TRAINED_MODELS_DIR = 'pre_trained_models'


def serialize_embeddings(model: Model, encoder, save_dir_path):
    embedding_layer: Layer = model.layers[1]
    weights = embedding_layer.get_weights()[0]

    vec_file_path = os.path.join(save_dir_path, VEC_FILE_NAME)
    vec_meta_file_path = os.path.join(save_dir_path, VEC_META_FILE_NAME)

    out_v = io.open(vec_file_path, 'w', encoding='utf-8')
    out_m = io.open(vec_meta_file_path, 'w', encoding='utf-8')

    for num, word in enumerate(encoder.tokens):
        vec = weights[num+1]
        out_m.write(word + '\n')
        out_v.write('\t'.join([str(x) for x in vec]) + '\n')

    out_v.close()
    out_m.close()
