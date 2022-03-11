import tensorflow.compat.v1 as tf
import profile_generator as pg
import pandas as pd
import preprocess.preprocess as preprocess
import random
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

keep_prob = 0.5

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def data_preprocess(X, Y):
    X = np.delete(X, range(4, X.shape[2]), 2)
    X = X.reshape(list(X.shape) + [1])
    Y = np.asarray(pd.cut(Y, bins=2, labels=[0, 1], right=False))
    Y = np.expand_dims(Y, axis=1)
    X = X.astype('float32')
    Y = Y.astype('float32')
    return X, Y

def test_sampler(methylations_test, annot_seqs_onehot, window_size, num_to_chr_dic):
    methylated, unmethylated = preprocess.methylations_subseter(methylations_test, window_size)
    test_sample_size = int(min(50000, 2*len(methylated), 2*len(unmethylated)))
    test_sample_set = methylated[:test_sample_size]+unmethylated[:test_sample_size]
    random.shuffle(test_sample_set)
    test_profiles, test_targets = pg.get_profiles(methylations_test, test_sample_set, [], annot_seqs_onehot, num_to_chr_dic, window_size=window_size)
    x_test, y_test = data_preprocess(test_profiles, test_targets)
    return x_test, y_test

def one_hot_encoder(data):
  values = np.array(data)
  label_encoder = LabelEncoder()
  integer_encoded = label_encoder.fit_transform(values.ravel())
  onehot_encoder = OneHotEncoder(sparse=False)
  integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
  onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
  return onehot_encoded

def net_MRCNN(x_fs):
    W_conv1 = weight_variable([1, 4, 1, 16])
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.conv2d(x_fs, W_conv1, strides=[1, 1, 4, 1], padding='VALID') + b_conv1
    h_conv1 = tf.reshape(h_conv1, [-1, 20, 20, 16])
    W_conv2 = weight_variable([3, 3, 16, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1,1,1,1], padding='VALID') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2,ksize=[1,3,3,1], strides=[1,3,3,1], padding='VALID')
    W_conv3 = weight_variable([3, 3, 32, 48])
    b_conv3 = bias_variable([48])
    h_conv3 = tf.nn.conv2d(h_pool2, W_conv3, strides=[1,1,1,1], padding='VALID')+ b_conv3
    W_conv4 = weight_variable([3, 3, 48, 64])
    b_conv4 = bias_variable([64])
    h_conv4 = tf.nn.conv2d(h_conv3, W_conv4, strides=[1,1,1,1], padding='VALID')+ b_conv4
    W_fc1 = weight_variable([2*2*64, 80])
    b_fc1 = bias_variable([80])
    h_pool4 = tf.reshape(h_conv4, [-1, 2*2*64])
    h_fc1 = tf.matmul(h_pool4, W_fc1) + b_fc1
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([80, 2])
    b_fc2 = bias_variable([2])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv

def run_experiment(methylations, sequences_onehot,num_to_chr_dic, data_size=500000):
    window_size = 400
    methylations_train, methylations_test = preprocess.seperate_methylations('', methylations, from_file=False)
    methylated_train, unmethylated_train = preprocess.methylations_subseter(methylations_train, window_size)
    print('experiment started')

    graph = tf.Graph()
    with graph.as_default():
        tf.disable_eager_execution()
        X_ph = tf.placeholder(tf.float32, shape=(None, 400, 4, 1), name='X')
        Y_ph = tf.placeholder(tf.float32, shape=(None, 2), name='Y')
        Z3 = net_MRCNN(X_ph)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_ph, logits=Z3))
        optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)
        x_test, y_test = test_sampler(methylations_test, sequences_onehot, window_size, num_to_chr_dic)
        y_test = one_hot_encoder(y_test)

    batch_size = 20
    epoc = 20
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        for i in range(epoc):
            for chunk in range(0, data_size, batch_size):
                if chunk+batch_size > data_size or chunk+batch_size > len(methylated_train) or chunk+batch_size > len(unmethylated_train):
                    continue
                else:
                    sample_set = methylated_train[chunk:chunk+batch_size]+unmethylated_train[chunk:chunk+batch_size]
                random.shuffle(sample_set)
                profiles, targets = pg.get_profiles(methylations_train, sample_set, sequences_onehot, [], num_to_chr_dic, window_size=window_size)
                X, Y = data_preprocess(profiles, targets)
                Y = one_hot_encoder(Y)
                feed_dict = {X_ph: X, Y_ph: Y}
                _, l = sess.run([optimizer, loss], feed_dict=feed_dict)
                if chunk % 100000 == 0:
                    print('it is running for ', str(chunk))
            if i%5 == 0:
                print('epoch ' + str(i))
        pred = Z3.eval({X_ph: x_test})
        return np.sum(np.argmax(pred, axis=1) == np.argmax(y_test, axis=1))/len(y_test)


