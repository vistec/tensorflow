# 输入文件队列
# tf.train.string_input_producer函数可将所有文件放入队列中
# 生成的输入队列可以同时被多个文件读取线程操作，输入队列可将队列
# 中的文件均匀的分给不同的线程
# 队列中的文件被处理完后，循环入列

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

files = tf.train.match_filenames_once("D:/AI/pyprogram/tensorflow/data/data.tfrecords-*")

filename_queue = tf.train.string_input_producer(files,shuffle=False,num_epochs=1)

reader = tf.TFRecordReader()

_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(
    serialized_example,
    features={
        'i':tf.FixedLenFeature([],tf.int64),
        'j':tf.FixedLenFeature([],tf.int64)
    }
)

with tf.Session() as sess:
    tf.local_variables_initializer().run()

    print(sess.run(files))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    for i in range(6):
        print(sess.run([features['i'],features['j']]))
        coord.request_stop()
        coord.join(threads)


