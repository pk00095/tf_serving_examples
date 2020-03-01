import time
import numpy as np
from PIL import Image
import cv2


import tensorflow as tf
import grpc
from tensorflow.contrib.util import make_tensor_proto

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


def run(image, model, host='localhost', port=8500, signature_name='serving_default'):

    channel = grpc.insecure_channel('{host}:{port}'.format(host=host, port=port))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    im = Image.open(image)
    data = np.array(im).astype(tf.keras.backend.floatx())
    # Read an image
    #data = imread(image)
    #data = data.astype(np.float32)
    #print(data)

    start = time.time()

    # Call classification model to make prediction on the image
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model
    request.model_spec.signature_name = signature_name
    request.inputs['input_image'].CopyFrom(make_tensor_proto(data, shape=[1, data.shape[0], data.shape[1], 3]))

    result = stub.Predict(request, 10.0)

    end = time.time()
    time_diff = end - start

    # Reference:
    # How to access nested values
    # https://stackoverflow.com/questions/44785847/how-to-retrieve-float-val-from-a-predictresponse-object
    #print(result)
    print('time elapased: {}'.format(time_diff))

    outputs_tensor_proto = result.outputs["predictions/Softmax:0"]
    shape = tf.TensorShape(outputs_tensor_proto.tensor_shape)
    #outputs = tf.constant(outputs_tensor_proto.float_val, shape=shape)
    outputs = np.array(outputs_tensor_proto.float_val).reshape(shape.as_list())

    #print(outputs)
    print(np.argmax(outputs), np.max(outputs))




if __name__ == '__main__':
    image_path = '/home/segmind/Downloads/doggy.jpeg'
    run(image_path, model='xception')