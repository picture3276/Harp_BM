#!/usr/bin/env python


import tensorflow as tf
import numpy as np
import time



def sample_prob(probs):
    return tf.nn.relu(
        tf.sign(
            probs - tf.random_uniform(tf.shape(probs))))



alpha = .01

W = tf.Variable(tf.random_normal([784, 100]), name="weights")
hb = tf.Variable(tf.zeros([100]), name="hbias")
vb = tf.Variable(tf.zeros([784]), name="vbias")
v0 = tf.placeholder("float", [None, 784])

h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
h0 = sample_prob(h0)
v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)
v1 = sample_prob(v1)
h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

pg = tf.matmul(tf.transpose(v0), h0)
ng = tf.matmul(tf.transpose(v1), h1)

dW = (pg - ng) / tf.to_float(tf.shape(v0)[0])
dhb = tf.reduce_mean(h0 - h1, 0)
dvb = tf.reduce_mean(v0 - v1, 0)

update_W_op = W.assign_add(alpha * dW)
update_hb_op = hb.assign_add(alpha * dhb)
update_vb_op = vb.assign_add(alpha * dvb)



with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    s = time.time()
    for _ in range(10):
        for start, end in zip(
                range(0, 50000, 100), range(100, 50000, 100)):
            X = np.random.random_sample((10, 784))

            _W = sess.run(update_W_op, feed_dict={v0: X})
            _hb = sess.run(update_hb_op, feed_dict={v0: X})
            _vb = sess.run(update_vb_op, feed_dict={v0: X})
    print(time.time()-s)
            
     





