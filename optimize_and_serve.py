import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph
import os, re, shutil
from glob import glob
from pprint import pprint

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from keras_retinanet import models
from keras_retinanet.utils.config import read_config_file, parse_anchor_parameters

def exporter(graph_def, export_path_base, sess):
    versions = []
    for i in glob(os.path.join(export_path_base,'*')):
        if os.path.isdir(i) and re.match(r'^[0-9]+$',i):
            versions.append(i)
    if versions == []:
        version = 0
    else:
        version = len(versions)

    export_path = os.path.join(export_path_base, str(version))
    os.makedirs(export_path)

    #with tf.Session() as sess:
        #with tf.gfile.GFile(graph_filepath, 'rb') as f:
        #    graph_def = tf.GraphDef()
        #    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    g_in = tf.import_graph_def(graph_def)
    tensor_input = sess.graph.get_tensor_by_name('import/input_1:0')
    classification_op = sess.graph.get_tensor_by_name('import/classification/concat:0')
    regression_op = sess.graph.get_tensor_by_name('import/regression/concat:0')

    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'input_image': tensor_input},
        outputs={'regression':regression_op, 'classification':classification_op})        
    #disable dropout and other train only ops
    #tf.keras.backend.set_learning_phase(0)

    print('exported model to {}'.format(export_path))

def get_graph_def_from_file(graph_filepath):
  with tf.Graph().as_default():
    with tf.gfile.GFile(graph_filepath, 'rb') as f:
      graph_def = tf.compat.v1.GraphDef()
      graph_def.ParseFromString(f.read())
      return graph_def

def optimize_graph(model_dir, graph_filename, transforms):
    input_names = ['input_1:0']
    output_names = ['regression/concat:0', 'classification/concat:0']
    optimized_graph_name = 'optimized_model.pb'
    if graph_filename is None:
        graph_def = get_graph_def_from_saved_model(model_dir)
    else:
        graph_def = get_graph_def_from_file(os.path.join(model_dir, graph_filename))

    optimized_graph_def = TransformGraph(
      graph_def,
      input_names,
      output_names,
      transforms)

    tf.io.write_graph(optimized_graph_def,
                      logdir=model_dir,
                      as_text=False,
                      name=optimized_graph_name)
    print('Graph optimized!')

    return os.path.join(model_dir, optimized_graph_name), optimized_graph_def




def freeze_model(model, session, export_path_base, graph_filename):

    export_path = export_path_base
    frozen_model_path = os.path.join(export_path, graph_filename)

    if os.path.isdir(export_path):
        shutil.rmtree(export_path)
    os.makedirs(export_path)

    #disable dropout and other train only ops
    #tf.keras.backend.set_learning_phase(0)

    #with tf.keras.backend.get_session() as sess:

    output_graph_def = tf.graph_util.convert_variables_to_constants(
        session,
        tf.get_default_graph().as_graph_def(),
        [t.op.name for t in model.outputs])

    with tf.io.gfile.GFile(frozen_model_path,'wb') as f:
        f.write(output_graph_def.SerializeToString())

    print('Find frozen graph at {}'.format(frozen_model_path))
    return frozen_model_path


def main():

    config_file = '/home/segmind/Desktop/test/tf-Graph_transform_tools/Howard_Hardhat-Retinanet_custom/config_3.ini'
    model_in = '/home/segmind/Desktop/test/tf-Graph_transform_tools/Howard_Hardhat-Retinanet_custom/resnet50_csv_10.h5'
    model_out_base = '/tmp/keras_retinanet'
    backbone = 'resnet50'
    graph_name = 'frozen_model.pb'

    args_config = read_config_file(config_file)
    anchor_parameters = parse_anchor_parameters(args_config)

    # load the model
    model = models.load_model(model_in, backbone_name=backbone)

    # convert the model
    print('Building inference model ..')
    model = models.convert_model(model, nms=True, class_specific_filter=True, anchor_params=anchor_parameters)

    tf.keras.backend.set_learning_phase(0)

    with tf.keras.backend.get_session() as sess:

        print('freezing model ..')
        frozen_graph = freeze_model(model, session=sess, export_path_base=model_out_base, graph_filename=graph_name)

        transforms = [
         'remove_nodes(op=Identity)', 
         'merge_duplicate_nodes',
         'strip_unused_nodes',
         'fold_constants(ignore_errors=true)',
         'fold_batch_norms',
         #'round_weights(num_steps=256)',
         #'quantize_weights'
        ]

        print('optimizing graph ..')
        optimized_graph, optimized_graph_def = optimize_graph(
          model_out_base, 
          graph_name, 
          transforms)

        print('Serving with tf-serving ..')

        exporter(
            graph_def=optimized_graph_def, 
            export_path_base=model_out_base,
            sess=sess)


if __name__ == '__main__':
    main()
