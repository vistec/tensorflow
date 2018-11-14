import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# F:\Deep_Learning\TFProjects\TensorflowProject\path\data_cat.tfrecords
# D:\AI\pyprogram\tensorflow\path

def read_and_decode_tfrecord(files):

    file_queue = tf.train.string_input_producer(files,shuffle=False)
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(file_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'img_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )

    label = tf.cast(features['label'], tf.int32)

    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image,[300, 300, 3])

    return image,label



files = tf.train.match_filenames_once(r"F:\Deep_Learning\TFProjects\TensorflowProject\path\traindata_pets.tfrecords-*")

image, label = read_and_decode_tfrecord(files)

image_batch, label_batch = tf.train.shuffle_batch(
    [image, label],
    batch_size=3,
    num_threads=3,
    capacity=1000,
    min_after_dequeue=30,
)

with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    cur_example_batch, cur_label_batch = sess.run(
        [image_batch, label_batch]
    )
    # b_image = Image.fromarray(cur_example_batch[0])

    plt.imshow(cur_example_batch[0])
    plt.axis('off')
    plt.show()
    plt.imshow(cur_example_batch[1])
    plt.axis('off')
    plt.show()
    plt.imshow(cur_example_batch[2])
    plt.axis('off')
    plt.show()

    coord.request_stop()
    coord.join(threads)
