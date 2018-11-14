# TFrecord数据准备

import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# F:\Deep_Learning\TFProjects\TensorflowProject\pic
cwd = r"D:\AI\pyprogram\tensorflow\pic"

pic_num = 0

record_file_num = 0
# D:\AI\pyprogram\tensorflow\path

file_path = r"D:\AI\pyprogram\tensorflow\path"

best_num = 1000
# 总共写入多少文件
num_examples = 1
# 每个文件多少数据
instances_per_shard = 2

batch_size = 1

classes = []

for i in os.listdir(cwd):
    classes.append(i)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image,max_delta=32. / 255.)
        image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
        image = tf.image.random_hue(image,max_delta=0.2)
        image = tf.image.random_contrast(image,lower=0.5,upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
        image = tf.image.random_brightness(image,max_delta=32. / 255.)
        image = tf.image.random_contrast(image,lower=0.5,upper=1.5)
        image = tf.image.random_hue(image,max_delta=0.2)
    elif color_ordering == 2:
        image = tf.image.random_hue(image,max_delta=0.2)
        image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
        image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
        image = tf.image.random_contrast(image,lower=0.5,upper=1.5)
    return tf.clip_by_value(image,0.0,1.0)

def preprocess_for_train(image,height,width):
    # if bbox is None:
    #     bbox = tf.constant(
    #         [0.0, 0.0, 1.0, 1.0],
    #         dtype=tf.float32,
    #         shape=[1,2,4]
    #     )

    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image,dtype=tf.float32)

    # bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image),bounding_boxes=bbox)
    #
    # distorted_image = tf.slice(image,bbox_begin,bbox_size)

    distorted_image = tf.image.resize_images(
        image,
        [height,width],
        method=1
    )

    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = distort_color(distorted_image,np.random.randint(3))

    return distorted_image




# F:\Deep_Learning\TFProjects\TensorflowProject\pic\cc-*
# D:\AI\pyprogram\tensorflow\pic

# https://www.jianshu.com/p/467cbc66875c?utm_source=oschina-app


if __name__ == '__main__':

    for index,name in enumerate(classes):
        class_path = os.path.join(cwd,name)

        for img_name in os.listdir(class_path):

            img_path = os.path.join(class_path,img_name)
            img_raw_data = tf.gfile.GFile(img_path,"rb").read()


            with tf.Session() as sess:
                # for i in range(3):
                for i in range(3):
                    img_data = tf.image.decode_jpeg(img_raw_data)
                    result = preprocess_for_train(img_data,300,300)

                    # img_data = tf.image.convert_image_dtype(result,dtype=tf.float32)
                    img_data = tf.image.encode_jpeg(img_data)

                    # img_name = img_path + "_edited_%d" % i
                    # with tf.gfile.GFile(img_name,"wb") as f:
                    #     f.write(img_data.eval())


                    plt.imshow(result.eval())
                    plt.show()


    # writer = tf.python_io.TFRecordWriter(os.path.join(file_path, "traindata_pets.tfrecords-000"))
    #
    # for index, name in enumerate(classes):
    #     class_path = os.path.join(cwd,name)
    #
    #     for img_name in os.listdir(class_path):
    #         pic_num = pic_num + 1
    #
    #         if pic_num > best_num:
    #             num = 1
    #             record_file_num += 1
    #             tfrecord_file_name = ("traindata_pets.tfrecords-%.3d" % record_file_num)
    #             writer = tf.python_io.TFRecordWriter(os.path.join(file_path,tfrecord_file_name))
    #
    #         img_path = os.path.join(class_path,img_name)
    #
    #         img_data = Image.open(img_path)
    #         img_data = img_data.resize([300,300])
    #         img_data = img_data.tobytes()
    #         example = tf.train.Example(
    #             features=tf.train.Features(
    #                 feature = {
    #                     'img_raw':_bytes_feature(img_data),
    #                     'label':_int64_feature(index),
    #                 }
    #             )
    #         )
    #
    #         writer.write(example.SerializeToString())
    # writer.close()

