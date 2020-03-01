import tensorflow as tf
import os, re
from glob import glob

#from tensorflow.keras.applications import Xception
#model = Xception(weights='imagenet', include_top=True)
def exporter(model, export_path_base):
	versions = []
	for i in glob(os.path.join(export_path_base,'*')):
		if os.isdir(i) and re.match(r'^[0-9]+$',i):
			versions.append(i)
	if versions == []:
		version = 0
	else:
		version = len(versions)

	export_path = os.path.join(export_path_base, str(version))
	os.makedirs(export_path)

	#disable dropout and other train only ops
	tf.keras.backend.set_learning_phase(0)

	with tf.keras.backend.get_session() as sess:
	    tf.saved_model.simple_save(
	        sess,
	        export_path,
	        inputs={'input_image': model.input},
	        outputs={t.name:t for t in model.outputs})

	print('exported model to {}'.format(export_path))