from PIL import Image
import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
import chineserecognition_bn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# temp_image = Image.open('boat.jpg').convert('L')
# temp_image = temp_image.resize((64, 64), Image.ANTIALIAS)
# temp_image = np.asarray(temp_image) / 255.0
# temp_image = temp_image.reshape([-1, 64, 64, 1])
#
# with tf.Session() as sess:
#     graph = chineserecognition_bn.build_graph(top_k=3)
#     saver = tf.train.Saver()
#     ckpt = tf.train.latest_checkpoint('../checkpoint/')
#     if ckpt:
#         saver.restore(sess, ckpt)
#     ans = sess.run([graph['images']], feed_dict={graph['images']: temp_image,
#                                               graph['keep_prob']: 1.0,
#                                               graph['is_training']: True})
#     print(ans.shape)

# import tensorflow as tf
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# a = tf.constant(3)
# with tf.Session() as sess:
#     sess.run(a)
#     print(a.eval)
#     b = tf.cast(a, tf.float32)
#     print(b.eval)

# a = '/../data/train/10001/'
# b = '../data/train/10000/'
# print('a>b:{}'.format(a > b))


a = tf.Variable([2.0, 3.0, 4.0, 5.0, 6.0, 7.0], tf.float32)
b = slim.dropout(a)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(b))
