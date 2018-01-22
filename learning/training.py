#!/usr/bin/env python3

import os
import sys
import random
import tensorflow as tf
import yaml
import re

from . import dataset

def save_yaml_model(sess, output_path, global_step):

    # Run tf to get all our variables
    variables = {v.name: sess.run(v) for v in tf.trainable_variables()}
    output = []

    # So we know when to move to the next list
    conv = -1
    layer = -1

    # Sorted so we see earlier layers first
    for k, v in sorted(variables.items()):
        key = re.match(r'Conv_(\d+)/Layer_(\d+)/(Weights|Biases):0', k)

        # If this is one of the things we are looking for
        if key is not None:
            c = int(key.group(1))
            l = int(key.group(2))
            var = key.group(3).lower()

            # If we change convolution add a new element
            if c != conv:
                output.append([])
                conv = c
                layer = -1

            # If we change layer add a new object
            if l != layer:
                output[-1].append({})
                layer = l

            if var not in output[conv][layer]:
                output[conv][layer][var] = v.tolist()
            else:
                raise Exception('Key already exists!')

    # Print as yaml
    with open(os.path.join(output_path, 'yaml_models', 'model_{}.yaml'.format(global_step)), 'w') as f:
        f.write(yaml.dump(output, width=120))


def build_training_graph(network, learning_rate):

    # The final expected output for the classes
    Y = network['Y']

    # The index of the final output we are checking
    Yi = network['Yi']

    # The unscaled network output
    logits = network['logits']

    # Gather our individual output for training
    with tf.name_scope('Training'):

        # Global training step
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        # Gather all our points with valid probabilities (the sum should always be 1.0)
        selected = tf.cast(tf.where(tf.reduce_sum(Y, axis=2) > 0.0), dtype=tf.int32)
        Yi = tf.gather_nd(Yi, selected)
        Y = tf.gather_nd(Y, selected)
        train_logits = tf.gather_nd(logits, tf.stack([selected[:,0], Yi], axis=1))

        # Our loss function
        with tf.name_scope('Loss'):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=train_logits, labels=Y, dim=1))

        # Our optimiser
        with tf.name_scope('Optimiser'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

        # Calculate accuracy
        with tf.name_scope('Metrics'):
            # Work out which class is larger and make 1 positive and 0 negative
            # TODO these need fixing for the multiple points per image
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

    for v in tf.trainable_variables():
        tf.summary.histogram(v.name, v)

    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    return optimizer, merged_summary_op, global_step

# Train the network
def train(sess,
          network,
          input_path,
          output_path,
          load_model=True,
          learning_rate=0.001,
          mesh_size=4,
          training_epochs=3,
          batch_size=100):

    # Build the training portion of the graph
    optimizer, merged_summary_op, global_step = build_training_graph(network, learning_rate)

    # Setup for tensorboard
    summary_writer = tf.summary.FileWriter(output_path, graph=tf.get_default_graph())

    # Create our model saver to save all the trainable variables and the global_step
    save_vars = {v.name: v for v in tf.trainable_variables()}
    save_vars.update({global_step.name: global_step})
    saver = tf.train.Saver(save_vars)

    # Initialise global variables
    sess.run(tf.global_variables_initializer())

    # Path to model file
    model_path = os.path.join(output_path, 'model.ckpt')

    # If we are loading existing training data do that
    if load_model and os.path.isfile(os.path.join(output_path, 'checkpoint')):
        print('Loading model {}'.format(model_path))
        saver.restore(sess, model_path)
    else:
        print('Creating new model {}'.format(model_path))

    # Load our dataset
    print('Loading file list')
    files = dataset.get_files(input_path, mesh_size)
    print('Loaded {} files'.format(len(files)))

    # First 80% for training
    train_dataset = dataset.dataset(
        files[:round(len(files) * 0.8)],
        variants=True,
        repeat=training_epochs,
        batch_size=20
    )

    # Last 20% for testing except for last 200 for validation
    test_dataset = dataset.dataset(
        files[round(len(files) * 0.8):-200],
        variants=False,
        repeat=1,
        batch_size=100
    )

    valid_dataset = dataset.dataset(
        files[-200:],
        variants=False,
        repeat=-1,
        shuffle=False,
        batch_size=200
    )

    # Get our handles
    train_handle, valid_handle = sess.run([train_dataset, valid_dataset])

    while True:
        try:
            # Run our training step
            sess.run([optimizer], feed_dict={
                network['handle']: train_handle
            })

            print("Batch:", tf.train.global_step(sess, global_step))

            # Every N steps do our validation/summary step
            if tf.train.global_step(sess, global_step) % 25 == 0:
                summary, = sess.run([merged_summary_op], feed_dict={
                    network['handle']: valid_handle
                })
                summary_writer.add_summary(summary, tf.train.global_step(sess, global_step))

            # Every N steps save our model
            if tf.train.global_step(sess, global_step) % 200 == 0:

                # Save the model after every pack
                saver.save(sess, model_path, tf.train.global_step(sess, global_step))

                # Save our model in yaml format
                save_yaml_model(sess, output_path, tf.train.global_step(sess, global_step))

        except tf.errors.OutOfRangeError:
            print("Training done")
            break
