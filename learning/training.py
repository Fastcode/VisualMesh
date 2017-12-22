#!/usr/bin/env python3

import os
import sys
import random
import tensorflow as tf
from PIL import Image, ImageDraw

from . import load

def build_training_graph(logits, learning_rate, beta):

    # The final expected output for the classes
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='Output')

    # The index of the final output we are checking
    Yi = tf.placeholder(dtype=tf.int32, shape=[None], name='OutputIndex')

    # Gather our individual output for training
    with tf.name_scope('Training'):

        # Get the indices to our selected objects
        training_indices = tf.bitcast(tf.stack([tf.bitcast(tf.range(tf.shape(Yi)[0], dtype=tf.int32), tf.float32),
                                                tf.bitcast(Yi, tf.float32)], axis=1), tf.int32)
        train_logits = tf.gather_nd(logits, training_indices)

        # Our loss function
        with tf.name_scope('Loss'):

            # Get our L2 regularisation information for weights
            l2_losses = []
            for var in tf.trainable_variables():
                if 'Weights' in var.name:
                    l2_losses.append(tf.nn.l2_loss(var))

            # Sum them up and multiply them by beta
            regularizers = beta * tf.add_n(l2_losses)

            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=train_logits, labels=Y, dim=1) + regularizers)

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

    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    return Y, Yi, optimizer, merged_summary_op

def save_validation_image(sess, network, v, output_cycle, validation_dir, output_dir):
    # Run our network for this validation object
    result = sess.run([network['network']], feed_dict={
        network['X']: [v['colours']],
        network['G']: [v['graph']],
        network['K']: 1.0 # don't dropout for images
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
    img.save(os.path.join(output_dir, '{:04d}_{:04d}.png'.format(v['file_no'] - 1, output_cycle)))


# Train the network
def train(sess, network, paths, load_model=False, learning_rate=0.001, training_epochs=2, regularisation=0.0, dropout=0.9, batch_size=1000):

    # Build the training portion of the graph
    Y, Yi, optimizer, merged_summary_op = build_training_graph(network['logits'], learning_rate, regularisation)

    # Setup for tensorboard
    summary_writer = tf.summary.FileWriter(paths['logs'], graph=tf.get_default_graph())

    # Create our model saver to save all the trainable variables
    saver = tf.train.Saver({v.name: v for v in tf.trainable_variables()})

    # Initialise global variables
    sess.run(tf.global_variables_initializer())

    # Path to model file
    model_path = os.path.join(paths['logs'], 'model.ckpt')

    # If we are loading existing training data do that
    if load_model:
        print('Loading model {}'.format(model_path))
        saver.restore(sess, model_path)
    else:
        print('Creating new model {}'.format(model_path))

    # Load our validation pack
    validation_pack = os.path.join(paths['validation'], 'validation.bin.lz4')
    sys.stdout.write('Loading validation pack {}...'.format(validation_pack));
    validation = load.validation(validation_pack)
    sys.stdout.write(' Loaded!\n');

    # List all our tree packs
    trees = sorted([os.path.join(paths['trees'], f) for f in os.listdir(paths['trees']) if f.endswith('.bin.lz4')])

    batch_no = 0
    output_cycle = 0

    # The number of epochs to train
    for epoch in range(training_epochs):

        # The rest of the trees for training
        for tree_i, tree in enumerate(trees):

            # Load the tree
            sys.stdout.write('Loading data tree pack {}...'.format(tree))
            data = load.tree(tree)
            sys.stdout.write(' Loaded!\n')

            # Get the next slice for training
            for i in range(0, len(data[0]), batch_size):

                # Run our training step
                _, summary = sess.run([optimizer, merged_summary_op], feed_dict={
                    network['X']: data[0][i:i + batch_size],
                    network['G']: data[3][i:i + batch_size],
                    network['K']: dropout,
                    Y: data[1][i:i + batch_size],
                    Yi: data[2][i:i + batch_size]
                })#, options=run_options, run_metadata=run_metadata)

                # Write summary log
                batch_no += 1
                summary_writer.add_summary(summary, batch_no)


                # tf.profiler.profile(tf.get_default_graph(),
                #                     run_meta=run_metadata,
                #                     options=(tf.profiler.ProfileOptionBuilder(tf.profiler.ProfileOptionBuilder.time_and_memory()).build()))


            # Save the model after every pack
            saver.save(sess, model_path, batch_no)

            # Every 5 packs save the images to show training progress
            if tree_i % 5 == 0:
                print('Saving images')
                # Run the network after each pack file for our example images
                output_cycle += 1
                for v in validation[:50:5]:
                    save_validation_image(sess, network, v, output_cycle, paths['validation'], paths['output_validation'])


        # Randomly shuffle the packs at the end of each epoch
        random.shuffle(trees)
