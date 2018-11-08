import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

files = tf.train.match_filenames_once('F:/Deep_Learning/TFProjects/TensorflowProject/path/data.tfrecords-*')
filename_queue = tf.train.string_input_producer(files,shuffle=False)

reader = tf.TFRecordReader()

_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'image'
    }
)