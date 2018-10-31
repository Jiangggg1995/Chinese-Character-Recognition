import pickle
import chineserecognition_bn
import os
import numpy as np
import tensorflow as tf
from PIL import Image


def test(path, images):
    with tf.Session() as sess:
        graph = chineserecognition_bn.build_graph(top_k=3)
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint('../checkpoint/')
        if ckpt:
            saver.restore(sess, ckpt)
        i = 1
        for image in images:
            image_path = path + image
            temp_image = Image.open(image_path).convert('L')
            temp_image = temp_image.resize((64, 64), Image.ANTIALIAS)
            temp_image = np.asarray(temp_image) / 255.0
            temp_image = temp_image.reshape([-1, 64, 64, 1])
            predict_val, predict_index = sess.run([graph['predicted_val_top_k'], graph['predicted_index_top_k']],
                                                  feed_dict={graph['images']: temp_image,
                                                             graph['keep_prob']: 1.0,
                                                             graph['is_training']: False})


with open('num_char', 'rb') as f:
    num_char = pickle.load(f)
# print(num_char)

path = 'C:\\Users\\Jiang\\Desktop\\chineserec\\newcut\\newcut_gender\\'
images = os.listdir(path)
# test(path=path, images=images)
with tf.Session() as sess:
    graph = chineserecognition_bn.build_graph(top_k=3)
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint('../checkpoint/')
    if ckpt:
        saver.restore(sess, ckpt)
    i = 1
    for image in images:
        image_path = path + image
        temp_image = Image.open(image_path).convert('L')
        temp_image = temp_image.resize((64, 64), Image.ANTIALIAS)
        temp_image = np.asarray(temp_image) / 255.0
        temp_image = temp_image.reshape([-1, 64, 64, 1])
        print(temp_image.shape)
        predict_val, predict_index = sess.run([graph['predicted_val_top_k'], graph['predicted_index_top_k']],
                                              feed_dict={graph['images']: temp_image,
                                                         graph['keep_prob']: 1.0,
                                                         graph['is_training']: False})
        character = num_char[predict_index[0][0]]
        print(image + ":" + character)
        i = i+1

