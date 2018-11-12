# TFrecord数据准备

import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 总共写入多少文件
num_examples = 1
# 每个文件多少数据
instances_per_shard = 2

batch_size = 1

image_raw_data = tf.gfile.FastGFile("F:\Deep_Learning\TFProjects\TensorflowProject\pic\cc.jpg",'rb').read()
img_xx_data = Image.open("F:\Deep_Learning\TFProjects\TensorflowProject\pic\cc.jpg")
img_xx_data.resize((300,300))
with tf.Session() as sess:

    img_data = tf.image.decode_jpeg(image_raw_data)

    # resize0 = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    img_data = tf.image.resize_images(img_data, [300, 300], method=1)

    # img_data = tf.cast(img_data,dtype=tf.int64)
    # reshape_img_data = tf.reshape(img_data,shape=[300*300*3])

    filename = ('F:/Deep_Learning/TFProjects/TensorflowProject/path/data_cat.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)

    example = tf.train.Example(features=tf.train.Features(
        feature={
            'image_raw':_int64_feature(0),
                # _bytes_feature(img_xx_data.tobytes()),
            'label':_int64_feature(1)
        }
    ))

    writer.write(example.SerializeToString())
    writer.close()

