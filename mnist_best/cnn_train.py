# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_best.cnn_inference as cnn_inference

BATCH_SIZE = 50
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "F:/Deep_Learning/TFProjects/TensorflowProject/mnist_best/pathtomodel/"
MODEL_NAME = "model.ckpt"

def train(mnist):
    x = tf.placeholder(tf.float32,[None,cnn_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, cnn_inference.OUTPUT_NODE], name='y-input')

    # regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    x_image = tf.reshape(x,[-1,28,28,1])
    y = cnn_inference.inference(x_image,train=True)
    global_step = tf.Variable(0,trainable=False)


    # variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    # variables_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    # cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # learning_rate = tf.train.exponential_decay(
    #     LEARNING_RATE_BASE,
    #     global_step,
    #     mnist.train.num_examples / BATCH_SIZE,
    #     LEARNING_RATE_DECAY
    # )

    train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy,global_step=global_step)

    # with tf.control_dependencies([train_step,variables_averages_op]):
    #     train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 精确度计算
    # saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            train_acc = accuracy.eval(feed_dict={x: xs, y_: ys})
            sess.run(train_step, feed_dict={x: xs, y_: ys})
            if i % 50 == 0:
                # _, loss_value, step, = sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
                print('step', i, 'training accuracy', train_acc)

        test_acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print("test accuracy", test_acc)
                # saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv = None):
    mnist = input_data.read_data_sets("F:/Deep_Learning/TFProjects/TensorflowProject/mnist_best/MINIST_data",one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()



