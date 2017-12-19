#!/usr/bin/env python3

import tensorflow as tf

import learning.load as load
from PIL import Image, ImageDraw
import os
import math

# Learning rate
learning_rate = 0.001

# Batch size
batch_size = 1000

# Number of times to train the network
training_epochs = 10

# Number of classes in the final output
n_classes = 2  # 2 classes, ball and not ball

# Number of neighbours for each point
graph_degree = 7

# Each number is output neurons for that layer, each list in the list is a convolution
groups = [[5, 3], [5, 3], [5, 3], [5, n_classes]]

# The mesh graph
G = tf.placeholder(dtype=tf.int32, shape=[None, None, graph_degree], name='MeshGraph')

# First input is the number of visual mesh points by 3
X = tf.placeholder(dtype=tf.float32, shape=[None, None, 3], name='Input')

# The final expected output for the classes
Y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='Output')

# The index of the final output we are checking
Yi = tf.placeholder(dtype=tf.int32, shape=[None], name='OutputIndex')

# Build our tensor
logits = X
for i, c in enumerate(groups):
    # Which convolution we are on
    with tf.variable_scope('Conv_{}'.format(i)):

        # The size of the previous output is the size of our input
        prev_last_out = logits.get_shape()[2].value

        with tf.variable_scope('GatherConvolution'):
            net_shape = tf.shape(logits)

            batch_idx = tf.bitcast(tf.reshape(tf.tile(tf.reshape(tf.range(net_shape[0], dtype=tf.int32), [-1, 1]), [1, net_shape[1] * graph_degree * prev_last_out]),
                                [net_shape[0], net_shape[1], graph_degree * prev_last_out]), tf.float32)

            neighbour_idx = tf.bitcast(tf.reshape(tf.tile(tf.reshape(G, [-1, 1]), [1, prev_last_out]),
                                [net_shape[0], net_shape[1], graph_degree * prev_last_out]), tf.float32)

            feature_idx = tf.bitcast(tf.reshape(tf.tile(tf.range(prev_last_out, dtype=tf.int32), [net_shape[0] * net_shape[1] * graph_degree]),
                               [net_shape[0], net_shape[1], graph_degree * prev_last_out]), tf.float32)

            # Now we can use this to lookup our actual network
            network_idx = tf.bitcast(tf.stack([batch_idx, neighbour_idx, feature_idx], axis=3, name='NetworkIndices'), tf.int32)
            logits = tf.gather_nd(logits, network_idx, name='NetworkGather')

        for j, out_s in enumerate(c):

            # Our input size is the previous output size
            in_s = logits.get_shape()[2].value

            # Which layer we are on
            with tf.variable_scope('Layer_{}'.format(j)):

                # Create weights and biases
                W = tf.get_variable('Weights', shape=[in_s, out_s], initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/in_s)), dtype=tf.float32)
                b = tf.get_variable('Biases', shape=[out_s], initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/out_s)), dtype=tf.float32)

                # Apply our weights and biases
                logits = tf.tensordot(logits, W, [[2], [0]], name="MatMul")
                logits = tf.add(logits, b)

                # Apply our activation function
                logits = tf.nn.elu(logits)


# Softmax our final output for all values in the mesh as our network
network = tf.nn.softmax(logits, name='Softmax', dim=2)

# Gather our individual output for training
with tf.name_scope('Training'):

    # Get the indices to our selected objects
    training_indices = tf.bitcast(tf.stack([tf.bitcast(tf.range(tf.shape(Yi)[0], dtype=tf.int32), tf.float32),
                                            tf.bitcast(Yi, tf.float32)], axis=1), tf.int32)
    train_logits = tf.gather_nd(logits, training_indices)

    # Our loss function
    with tf.name_scope('Loss'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=train_logits, labels=Y, dim=1))

    # Our optimiser
    with tf.name_scope('Optimiser'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Calculate accuracy
    with tf.name_scope('Metrics'):
        # Work out which class is larger and make 1 positive and 0 negative
        softmax_logits = tf.nn.softmax(train_logits)
        predictions = tf.argmin(softmax_logits, axis=1)
        labels = tf.argmin(Y, axis=1)

        # Get our confusion matrix
        tp = tf.cast(tf.count_nonzero(predictions * labels), tf.float32)
        tn = tf.cast(tf.count_nonzero((predictions - 1) * (labels - 1)), tf.float32)
        fp = tf.cast(tf.count_nonzero(predictions * (labels - 1)), tf.float32)
        fn = tf.cast(tf.count_nonzero((predictions - 1) * labels), tf.float32)

        # Calculate our confusion matrix
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)
        acc = (tp + tn) / (tp + tn + fp + fn)
        f1 = 2.0 * ((ppv * tpr) / (ppv + tpr))
        mcc = (tp * tn - fp * fn) / tf.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        # How wide the gap is between the two classes
        certainty = tf.reduce_mean(tf.abs(tf.subtract(softmax_logits[:, 0], softmax_logits[:, 1])))

# Monitor cost and metrics
tf.summary.scalar('Loss', cost)
tf.summary.scalar('True Positive Rate', tpr)
tf.summary.scalar('True Negative Rate', tnr)
tf.summary.scalar('Positive Predictive', ppv)
tf.summary.scalar('Negative Predictive', npv)
tf.summary.scalar('Accuracy', acc)
tf.summary.scalar('F1 Score', f1)
tf.summary.scalar('Matthews', mcc)
tf.summary.scalar('Certainty', certainty)

# Create summaries to visualize weights
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)

# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# Initialise variables
init_op = tf.global_variables_initializer()

# We save our trainable variables
saver = tf.train.Saver({v.name: v for v in tf.trainable_variables()})

config = tf.ConfigProto()
config.graph_options.build_cost_model = 1
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
config.gpu_options.force_gpu_compatible = True

run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

with tf.Session(config=config) as sess:

    # Initialise variables
    sess.run(init_op)

    # Checkpoint path
    model_path = os.path.join('training', 'model.ckpt')

    # Load model or not
    load_model = True
    if load_model:
        saver.restore(sess, model_path)

    # Setup for tensorboard
    summary_writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())

    # Grab our training tree packs
    tree_dir = os.path.join('training', 'trees')
    trees = sorted([os.path.join(tree_dir, f) for f in os.listdir(tree_dir) if f.endswith('.bin.gz')])

    # Grab our validation pack
    validation_dir = os.path.join('training', 'validation')
    validation_pack = os.path.join(validation_dir, 'validation.bin.gz')
    print('Loading validation treepack')
    validation = load.validation(validation_pack)
    print('\tloaded')

    training_samples = 0
    output_file_no = 0

    # The number of epochs to train
    for epoch in range(training_epochs):

        # The rest of the trees for training
        for tree_i, tree in enumerate(trees):

            # Load the tree
            print('Loading data tree pack {}'.format(tree))
            data = load.tree(tree)
            print('\tloaded')

            # Get the next slice for training
            for i in range(0, len(data[0]), batch_size):

                # Run our training step
                _, c, summary = sess.run([optimizer, cost, merged_summary_op], feed_dict={
                    X: data[0][i:i + batch_size],
                    Y: data[1][i:i + batch_size],
                    Yi: data[2][i:i + batch_size],
                    G: data[3][i:i + batch_size]
                })#, options=run_options, run_metadata=run_metadata)

                # Write summary log
                training_samples += len(data[0][i:i + batch_size])
                summary_writer.add_summary(summary, training_samples)

                # tf.profiler.profile(tf.get_default_graph(),
                #                     run_meta=run_metadata,
                #                     options=(tf.profiler.ProfileOptionBuilder(tf.profiler.ProfileOptionBuilder.time_and_memory()).build()))

            # Save the model after each pack
            saver.save(sess, model_path)

            # Every 5 packs save the images to show training progress
            if tree_i % 5 == 0:
                # Run the network after each pack file for our example images
                output_file_no += 1
                for v in validation[:50:5]:

                    # Run our network for this validation object
                    result = sess.run([network], feed_dict={
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

                                d.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=(r, g, 0, 255))

                    # Save our image
                    img.save(os.path.join('output', 'validation', '{:04d}_{:04d}.png'.format(v['file_no'] - 1, output_file_no)))
