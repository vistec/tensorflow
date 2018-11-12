import tensorflow as tf
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

image_raw_data = tf.gfile.FastGFile("F:\Deep_Learning\TFProjects\TensorflowProject\pic\cc.jpg",'rb').read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)

    resize0 = tf.image.convert_image_dtype(img_data,dtype=tf.float32)

    img_data = tf.image.resize_images(resize0,[300,300],method=0)



    batched = tf.expand_dims(
        tf.image.convert_image_dtype(img_data, tf.float32),
        0
    )

    boxes = tf.constant([[[0.05,0.5,0.9,0.7]]])
    result = tf.image.draw_bounding_boxes(batched,boxes)

    plt.figure(1)
    plt.imshow(result.eval().reshape([300,300,3]))
    plt.show()



    # img_data = tf.image.convert_image_dtype(resized, dtype=tf.uint8)
    #
    # encoded_img = tf.image.encode_jpeg(img_data)
    #
    # with tf.gfile.FastGFile("ss.jpg",'wb') as f:
    #     f.write(encoded_img.eval())







