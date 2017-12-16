#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import gzip
import struct
import os
import re


def load_pack(file):

    # The 200 point visual mesh input
    X = []

    # The true/falseness of the check point
    Y = []

    # The index of the check point
    Yi = []

    # The adjacency graph
    G = []

    with gzip.open(file, 'rb') as f:

        while True:

            # Try to read 8 bytes
            d1 = f.read(8)
            if len(d1) < 8:
                break

            # The index of the check and the value
            ci, cv = struct.unpack('<If', d1)

            # Graph data
            graph = np.frombuffer(f.read(200 * 4 * 7), np.int32)

            # Colour data
            colours = np.frombuffer(f.read(200 * 4 * 3), np.float32).reshape(3, -1, order='F')
            stencil = np.frombuffer(f.read(200 * 4 * 1), np.float32)

            # Append our various datas to the list
            X.append(colours)
            Y.append(np.array([cv, 1.0 - cv]))
            Yi.append(ci)
            G.append(graph)

    # Squash the results into numpy arrays and return
    return np.array(X), np.array(Y), np.array(Yi), np.array(G)


# Learning rate
learning_rate = 0.001

# Number of classes in the final output
n_classes = 2  # 2 classes, ball and not ball

# Number of neighbours for each point
graph_degree = 7

# Each number is output neurons for that layer, each list in the list is a convolution
groups = [[5, 3], [5, 3], [5, 3], [5, n_classes]]

# The mesh graph
G = tf.placeholder(tf.int32, [None, 200 * graph_degree], name="MeshGraph")
onehot_G = tf.one_hot(G, 200)

# First input is the number of visual mesh points by 3
X = tf.placeholder(tf.float32, [None, 3, 200], name="Input")

# The final expected output for the classes
Y = tf.placeholder(tf.float32, [None, 2], name="Output")

# The index of the final output we are checking
Yi = tf.placeholder(tf.int32, [None, 1], name="OutputIndex")


# Build our tensor
logits = X
count = 0
for p, c in zip([[X.get_shape()[1].value]] + groups, groups):
    count += 1
    prev_last_out = p[-1]

    # Gather the relevant rows and reshape to stack them
    logits = tf.einsum('ajk,aik->aij', logits, onehot_G)
    logits = tf.reshape(logits, [-1, prev_last_out * graph_degree, 200], name="Reshape" + str(count))

    for in_s, out_s in zip([prev_last_out * graph_degree] + c, c):
        # Apply weights and biases
        W = tf.Variable(tf.random_normal([in_s, out_s]))
        b = tf.Variable(tf.random_normal([out_s]))

        logits = tf.matrix_transpose(tf.tensordot(logits, W, [[1], [0]], name="MatMul1") + b)
        # Apply our activation function
        logits = tf.nn.elu(logits)

logits = tf.nn.softmax(logits, dim=1)
logits = tf.einsum('ajk,aik->aj', logits, tf.one_hot(Yi, 200))

with tf.name_scope('Loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

with tf.name_scope('Optimiser'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.name_scope('Accuracy'):
    acc = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

# Monitor cost and accuracy tensor
tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', acc)

# Create summaries to visualize weights
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)

# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:

    batch_size = 1000
    training_epochs = 10

    # Initialise the variables
    sess.run(tf.global_variables_initializer())

    # Setup for tensorboard
    summary_writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())

    # Grab our training packs
    pack_dir = os.path.join('training', 'trees')
    packs = sorted([os.path.join(pack_dir, f) for f in os.listdir(pack_dir) if f.endswith('.bin.gz')])

    # First pack will be used as validation data
    print('Loading pack {} as validation'.format(packs[0]))
    validation = load_pack(packs[0])
    print('\tloaded')

    training_samples = 0

    # The number of epochs to train
    for epoch in range(training_epochs):

        # The rest of the packs for training
        for pack in packs[1:]:

            # Load the pack
            print('Loading data pack {}'.format(pack))
            data = load_pack(pack)
            print('\tloaded')

            # Get the next slice for training
            for i in range(0, len(data[0]), batch_size):

                # Run our training step
                _, c, summary = sess.run([optimizer, cost, merged_summary_op], feed_dict={
                    X: data[0][i:i + batch_size],
                    Y: data[1][i:i + batch_size],
                    Yi: np.array(data[2][i:i + batch_size]).reshape((-1, 1)),
                    G: data[3][i:i + batch_size]
                })

                # Write summary log
                training_samples += len(data[0][i:i + batch_size])
                summary_writer.add_summary(summary, training_samples)
