import tensorflow as tf
from tensorflow.keras.models import Model


def show_results(model: Model, dataset: tf.data.Dataset, encoder):
    padded_dataset = dataset.padded_batch(64, padded_shapes=([-1], []))
    unbatched_dataset = padded_dataset.unbatch()

    predictions = model.predict(padded_dataset)

    results = []
    for index, sample in enumerate(unbatched_dataset):
        item = [encoder.decode(sample[0].numpy()), sample[1].numpy(), predictions[index].item()]
        results.append(item)

    return results
