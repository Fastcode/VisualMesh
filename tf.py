#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import gzip
import struct
from PIL import Image, ImageDraw
import os


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
            graph = np.frombuffer(f.read(200 * 4 * 7), np.int32).reshape(-1, 7, order='C')

            # Colour data
            colours = np.frombuffer(f.read(200 * 4 * 3), np.float32).reshape(-1, 3, order='C')
            stencil = np.frombuffer(f.read(200 * 4 * 1), np.float32)

            # Append our various datas to the list
            X.append(colours)
            Y.append(np.array([cv, 1.0 - cv]))
            Yi.append(ci)
            G.append(graph)

    # Squash the results into numpy arrays and return
    return np.array(X), np.array(Y), np.array(Yi), np.array(G)


def load_validation(file):

    # THe output images
    output = []

    # Open the file
    with gzip.open(file, 'rb') as f:

        while True:

            # Try to read 8 bytes and end when there are none
            d1 = f.read(8)
            if len(d1) < 8:
                break

            # The file number and the size of this mesh
            file_no, size = struct.unpack('<II', d1)

            # Graph data (7 values per point)
            graph = np.frombuffer(f.read(size * 4 * 7), np.int32).reshape(-1, 7, order='C').astype(np.float32)

            # Pixel coordinates (2 values per point)
            coords = np.frombuffer(f.read(size * 4 * 2), np.int32).reshape(-1, 2, order='C')

            # Colour data (3 values per point)
            colours = np.frombuffer(f.read(size * 4 * 3), np.float32).reshape(-1, 3, order='C')

            # Stencil data
            stencil = np.frombuffer(f.read(size * 4 * 1), np.float32)

            output.append({
                'file_no': file_no,
                'graph': graph,
                'coords': coords,
                'colours': colours,
                'stencil': stencil
            })

    # Squash the results into numpy arrays and return
    return output


# Learning rate
learning_rate = 0.001

# Batch size
batch_size = 1000

# Number of classes in the final output
n_classes = 2  # 2 classes, ball and not ball

# Number of neighbours for each point
graph_degree = 7

# Each number is output neurons for that layer, each list in the list is a convolution
groups = [[5, 3], [5, 3], [5, 3], [5, n_classes]]

# The size of the mesh graph
Gs = tf.placeholder(dtype=tf.int32, name='MeshSize')

# The mesh graph
G = tf.placeholder(dtype=tf.int32, shape=[None, None, graph_degree], name='MeshGraph')


# First input is the number of visual mesh points by 3
X = tf.placeholder(dtype=tf.float32, shape=[None, None, 3], name='Input')

# The final expected output for the classes
Y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='Output')

# The index of the final output we are checking
Yi = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='OutputIndex')

# Build our tensor
network = X
for i, c in enumerate(groups):
    # Which convolution we are on
    with tf.name_scope('Conv_{}'.format(i)):

        # The size of the previous output is the size of our input
        prev_last_out = network.get_shape()[2].value

        with tf.name_scope('GatherConvolution'):

            if False:
                # Construct a 4D tensor of size [?, 200, graph_degree * prev_last_output, 3] filled with a grid of indices
                net_shape = tf.shape(network)
                idx_a, idx_b, idx_c = tf.meshgrid(tf.range(net_shape[0]),
                                                  tf.range(net_shape[1]),
                                                  tf.range(graph_degree * prev_last_out),
                                                  indexing='ij',
                                                  name='Indices')

                # Use these indexes to lookup our graph indices
                # TODO note we have to hack using bitcast as tensorflow won't gather int32s on the GPU FOR NO RAISIN
                idx_b = tf.bitcast(tf.gather_nd(tf.bitcast(G, type=tf.float32), tf.stack([idx_a, idx_b, tf.div(idx_c, graph_degree)], axis=3), name='GraphIndices'), type=tf.int32)

                # Now we can use this to lookup our actual network
                network = tf.gather_nd(network, tf.stack([idx_a, idx_b, tf.mod(idx_c, prev_last_out)], axis=3), name='NetworkIndices')

            else:
                # create a [None,200,graph_degree,200] tensor of one-hot vectors to select the correct indices of G
                graph_indices = tf.one_hot(G, Gs)
                network = tf.einsum("ijk,ibaj->ibak", network, graph_indices)
                network = tf.reshape(network, [-1, 200, prev_last_out * graph_degree])

        for j, out_s in enumerate(c):

            # Our input size is the previous output size
            in_s = network.get_shape()[2].value

            # Which layer we are on
            with tf.name_scope('Layer_{}'.format(j)):

                # Create weights and biases
                W = tf.Variable(tf.truncated_normal(shape=[in_s, out_s], mean=0.0), dtype=tf.float32, name='Weights')
                b = tf.Variable(tf.truncated_normal(shape=[out_s], mean=0.0), dtype=tf.float32, name='Biases')

                if False:
                    # Multiply each slice by the weights matrix
                    network = tf.einsum('ijk,kl->ijl', network, W)

                    # Add the bias
                    network = tf.add(network, b)
                else:
                    network = tf.tensordot(network, W, [[2], [0]], name="MatMul1") + b

                # Apply our activation function
                network = tf.nn.elu(network)

# Softmax our final output for all values in the mesh
network = tf.nn.softmax(network, name='Softmax')

# Gather our individual output for training
with tf.name_scope('Training'):
    training_indices = tf.one_hot(indices=Yi, depth=Gs, dtype=tf.float32, name='OneHotTrainingIndices')
    train_logits = tf.einsum('ijk,ibj->ibk', network, training_indices)
    train_logits = tf.reshape(train_logits, [-1, 2])

    # Our loss function
    with tf.name_scope('Loss'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=train_logits, labels=Y))

    # Our optimiser
    with tf.name_scope('Optimiser'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Calculate accuracy
    with tf.name_scope('Accuracy'):
        acc = tf.equal(tf.argmax(train_logits, 1), tf.argmax(Y, 1))
        acc = tf.reduce_mean(tf.cast(acc, tf.float32))

# Monitor cost and accuracy tensor
tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', acc)

# Create summaries to visualize weights
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)

# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
with tf.Session(config=config) as sess:

    training_epochs = 10

    # Initialise the variables
    sess.run(tf.global_variables_initializer())

    # Setup for tensorboard
    summary_writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())

    # Grab our training packs
    pack_dir = os.path.join('training', 'trees')
    packs = sorted([os.path.join(pack_dir, f) for f in os.listdir(pack_dir) if f.endswith('.bin.gz')])

    # First pack will be used as validation data
    validation_dir = os.path.join('training', 'validation')
    validation_pack = os.path.join(validation_dir, 'validation.bin.gz')
    print('Loading validation pack')
    validation = load_validation(validation_pack)
    print('\tloaded')

    training_samples = 0
    output_file_no = 0

    # The number of epochs to train
    for epoch in range(training_epochs):

        # The rest of the packs for training
        for pack in packs:

            # Load the pack
            print('Loading data pack {}'.format(pack))
            data = load_pack(pack)
            print('\tloaded')

            # Get the next slice for training
            for i in range(0, len(data[0]), batch_size):

                # Run our training step
                _, c, summary = sess.run([optimizer, cost, merged_summary_op], feed_dict={
                    Gs: 200,
                    X: data[0][i:i + batch_size],
                    Y: data[1][i:i + batch_size],
                    Yi: np.array(data[2][i:i + batch_size]).reshape((-1, 1)),
                    G: data[3][i:i + batch_size]
                })

                # Write summary log
                training_samples += len(data[0][i:i + batch_size])
                summary_writer.add_summary(summary, training_samples)

            continue

            # Run the network after each pack file for our example images
            output_file_no += 1
            for v in validation:

                # Run our network for this validation object
                result = sess.run([network], feed_dict={
                    Gs: len(v['colours']),
                    X: [v['colours']],
                    G: [v['graph']]
                })

                # Load our input image
                img = Image.open(os.path.join(validation_dir, 'image{:07d}.png'.format(v['file_no'] - 1)))
                d = ImageDraw.Draw(img)

                # Go through our drawing coordinates
                for i in range(len(v['coords'])):

                    # The result at our point
                    r1 = result[0][0][i]

                    # The pixel coordinates at our point
                    p1 = v['coords'][i]

                    # Go through our neighbours
                    for n in v['graph'][i]:

                        # The result at our neighbours point
                        r2 = result[0][0][i]

                        # The coordinate at our neighbours point
                        p2 = v['coords'][n]

                        # Draw a line if both are in the image
                        if p2[0] != -1 and p2[1] != -1:
                            r = max(min(int(round(r1[1] * 255)), 255), 0)
                            g = max(min(int(round(r1[0] * 255)), 255), 0)

                            d.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=(r, g, 0, 64))

                # Save our image
                img.save(os.path.join('output', 'validation', '{:04d}_{:04d}.png'.format(v['file_no'], output_file_no)))
