# QueueRunner 用于启动多个线程来操作同一个队列，启动的这些线程可以通过tf.
# Coordinator类来统一管理
#

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

queue = tf.FIFOQueue(100,'float')
enqueue_op = queue.enqueue([tf.random_normal([1])])

qr = tf.train.QueueRunner(queue,[enqueue_op] * 5)
tf.train.add_queue_runner(qr)


out_tensor = queue.dequeue()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    for _ in range(10):
        print(sess.run(out_tensor)[0])


    coord.request_stop()
    coord.join()
