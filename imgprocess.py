import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


catfile_queue = tf.train.string_input_producer([r"F:\Deep_Learning\TFProjects\TensorflowProject\path\data_cat.tfrecords"])
files = tf.train.match_filenames_once(r"F:\Deep_Learning\TFProjects\TensorflowProject\path\data_cat.tfrecords")

filename_queue = tf.train.string_input_producer(files,shuffle=False)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw':tf.FixedLenFeature([],tf.string),
        'label':tf.FixedLenFeature([],tf.int64)
    }
)

image = tf.decode_raw(features['image_raw'],tf.uint8)

label = tf.cast(features['label'],tf.int32)

image = tf.reshape(image, [300, 300, 3])

image_batch, label_batch = tf.train.shuffle_batch(
    [image,label],
    batch_size=1,
    capacity=100,
    min_after_dequeue=30
)

with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    cur_example_batch, cur_label_batch = sess.run(
        [image, label]
    )
    print(cur_example_batch, cur_label_batch)

    coord.request_stop()
    coord.join(threads)