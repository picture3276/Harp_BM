import tensorflow as tf
import numpy as np
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # compress warning

# global config
N = 1000
Iter = 200


np.random.seed(1)
X_ = np.ones((N, N))
D_ = np.eye(N)


# for i < Iter:
#     A = D * (X - D' * A)
def graph_orig(Iter, wg=False):

    g1 = tf.Graph()

    with g1.as_default():

        with tf.variable_scope("g1"):
            A = tf.Variable(tf.zeros(shape=[N, N]), dtype=tf.float32, name='A')
            D = tf.placeholder(shape=[N, N], dtype=tf.float32, name='D')
            X = tf.placeholder(shape=[N, N], dtype=tf.float32, name='X')

            R = tf.matmul(D, tf.subtract(X, tf.matmul(tf.transpose(D), A)), name="DxX-DTA")
            L = tf.assign(A, R, name='result')

            # tf.train.write_graph(tf.get_default_graph(), os.getcwd(), 'graph_org.json')
            # tf.train.write_graph(tf.get_default_graph(), os.getcwd(), 'graph_org.txt')
            # tf.summary.FileWriter('./graphs', tf.get_default_graph())
            
            init = tf.global_variables_initializer()

            with tf.Session() as sess:
                # if wg:
                #     writer = tf.summary.FileWriter('./g/g1', sess.graph)
                #     writer.add_graph(sess.graph)

                sess.run(init)

                s = time.time()
                for i in range(Iter):
                    result = sess.run(L, feed_dict={D: D_, X: X_})
                print(time.time() - s)

                # print(result)

                # if wg:
                #     writer.close()


graph_orig(Iter)


