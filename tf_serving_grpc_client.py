import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import time, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import grpc
from tensorflow.contrib.util import make_tensor_proto

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from keras_retinanet.utils.image import resize_image

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


font = ImageFont.load_default()

def run(image, model, host='localhost', port=8500, signature_name='serving_default'):

    channel = grpc.insecure_channel('{host}:{port}'.format(host=host, port=port))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    im = Image.open(image)
    width, height = im.size
    im = im.resize(size=(min(width,1333), min(800,height)))
    data = np.array(im).astype(tf.keras.backend.floatx())

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

    bboxes_proto = result.outputs["filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0"]
    bboxes_proto_shape = tf.TensorShape(bboxes_proto.tensor_shape)
    bboxes = tf.constant(bboxes_proto.float_val, shape=bboxes_proto_shape) #*scale


    confidences_proto = result.outputs["filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0"]
    confidences_proto_shape = tf.TensorShape(confidences_proto.tensor_shape)
    confidences = tf.constant(confidences_proto.float_val, shape=confidences_proto_shape)

    labels_proto = result.outputs["filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0"]
    labels_shape = tf.TensorShape(labels_proto.tensor_shape)
    labels = tf.constant(labels_proto.int_val, shape=labels_shape)

    return bboxes.numpy(), confidences.numpy(), labels.numpy()


def annotate_image(image_path, bboxes, scores, labels):

    uniquelabels = list()
    labelcolour = list()
    im = Image.open(image_path)
    draw = ImageDraw.Draw(im)

    for bbox, confidence, label in zip(bboxes, scores, labels):
        if label < 0 or confidence <0.5:
            break

        if label not in uniquelabels:
            colour = random.sample(STANDARD_COLORS,1)[-1]
            while colour in labelcolour:
                colour = random.sample(STANDARD_COLORS,1)[-1]
            #uniquelabels[label] = labelcolour
            uniquelabels.append(label)
            labelcolour.append(colour)

        colortofill = labelcolour[uniquelabels.index(label)]

        xmin, ymin, xmax, ymax = bbox

        draw.rectangle([xmin,ymin,xmax,ymax], fill=None, outline=colortofill)

        display_label = '{}|{:.2f}%'.format(label, confidence*100)
        display_str_heights = font.getsize(display_label)[1]

        total_display_str_height = (1 + 2 * 0.05) * display_str_heights

        if ymin > total_display_str_height:
            text_bottom = ymin
        else:
            text_bottom = ymax + total_display_str_height

        text_width, text_height = font.getsize(display_label)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([xmin, text_bottom-text_height-2*margin, xmin+text_width, ymin], fill=colortofill)

        #draw.text([xmin,ymin], label, fill='white', font=ImageFont.load_default())
        draw.text((xmin+margin, text_bottom-text_height-margin),display_label,fill='black',font=font)

    im.show()





if __name__ == '__main__':
    image_path = '/home/segmind/Desktop/test/tfdv/HardHat/Hardhat/Test/JPEGImage/005313.jpg'
    bboxes, scores, labels = run(image_path, model='retinanet')

    print('annotating image ..')
    annotate_image(image_path, bboxes[0], scores[0], labels[0])
