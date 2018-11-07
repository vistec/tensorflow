# TFrecord数据准备

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 总共写入多少文件
num_shards = 2
# 每个文件多少数据
instances_per_shard = 2

for i in range(num_shards):
    filename = ('D:/AI/pyprogram/tensorflow/data/data-%.5d-of-%.5d' % (i,num_shards))
    writer = tf.python_io.TFRecordWriter(filename)

    for j in range(instances_per_shard):
        example = tf.train.Example(features=tf.train.Features(feature={
            'i':_int64_feature(i),
            'j':_int64_feature(j)
        }))

        writer.write(example.SerializeToString())
    writer.close()

