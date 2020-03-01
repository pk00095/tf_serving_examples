import tensorflow as tf
from tensorflow.keras.applications import Xception
import os

tf.keras.backend.set_learning_phase(0)
model = Xception(weights='imagenet', include_top=True)

version = 2
export_path = '/tmp/classification/{}'.format(int(version))
os.makedirs(export_path, exist_ok=True)

# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors
# And stored with the default serving key
with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'input_image': model.input},
        outputs={t.name:t for t in model.outputs})