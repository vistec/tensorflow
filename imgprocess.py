import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# def distort_color(image, color_ordering=0):
#     if color_ordering == 0:
#         image = tf.image.random_brightness(image,max_delta=32. / 255.)
#         image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
#         image = tf.image.random_hue(image,max_delta=0.2)
#         image = tf.image.random_contrast(image,lower=0.5,upper=1.5)
#     elif color_ordering == 1:
#         image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
#         image = tf.image.random_brightness(image,max_delta=32. / 255.)
#         image = tf.image.random_contrast(image,lower=0.5,upper=1.5)
#         image = tf.image.random_hue(image,max_delta=0.2)
#     elif color_ordering == 2:
#         image = tf.image.random_hue(image,max_delta=0.2)
#         image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
#         image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
#         image = tf.image.random_contrast(image,lower=0.5,upper=1.5)
#     return tf.clip_by_value(image,0.0,1.0)
#
# def preprocess_for_train(image,height,width,bbox):
#     if bbox is None:
#         bbox = tf.constant(
#             [0.0, 0.0, 1.0, 1.0],
#             dtype=tf.float32,
#             shape=[1,2,4]
#         )
#
#     if image.dtype != tf.float32:
#         image = tf.image.convert_image_dtype(image,dtype=tf.float32)
#
#     bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image),bounding_boxes=bbox)
#
#     distorted_image = tf.slice(image,bbox_begin,bbox_size)
#
#     distorted_image = tf.image.resize_images(
#         distorted_image,
#         [height,width],
#         method=1
#     )
#
#     distorted_image = tf.image.random_flip_left_right(distorted_image)
#     distorted_image = distort_color(distorted_image,np.random.randint(3))
#
#     return distorted_image


catfile_queue = tf.train.string_input_producer([r"D:\AI\pyprogram\tensorflow\path\data_cat.tfrecords"])
files = tf.train.match_filenames_once(r"D:\AI\pyprogram\tensorflow\path\data_cat.tfrecords")

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

image = tf.reshape(image, tf.stack([300, 300, 3]))

print(tf.shape(image))


# image_batch, label_batch = tf.train.shuffle_batch(
#     [image,label],
#     batch_size=1,
#     capacity=100,
#     min_after_dequeue=30
# )
#
# with tf.Session() as sess:
#
#
#     init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
#     sess.run(init_op)
#
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#     cur_example_batch, cur_label_batch = sess.run(
#         [image, label]
#     )
#     print(cur_example_batch, cur_label_batch)
#
#     coord.request_stop()
#     coord.join(threads)

