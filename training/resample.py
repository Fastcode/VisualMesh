#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf

from . import dataset

def resample(sess,
             network,
             mesh_type,
             mesh_size,
             model_dir,
             input_path,
             output_path):

    # Make our output directory
    os.makedirs(output_path, exist_ok=True)

    # Initialise global variables
    sess.run(tf.global_variables_initializer())

    save_vars = {v.name: v for v in tf.trainable_variables()}
    saver = tf.train.Saver(save_vars)

    # Get our model directory and load it if it exists
    model_path = os.path.join(model_dir, 'model.ckpt')
    checkpoint_file = tf.train.latest_checkpoint(model_dir)
    print('Loading model {}'.format(checkpoint_file))
    saver.restore(sess, checkpoint_file)

    # Load our dataset
    print('Loading file list')
    files = dataset.get_files(input_path, mesh_type, mesh_size)
    print('Loaded {} files'.format(len(files)))

    # Load all our data
    data_string = dataset.dataset(
        files,
        mesh_type=mesh_type,
        variants=False,
        repeat=1,
        batch_size=1,
        shuffle=False
    )

    # Get our iterator handle
    data_handle = sess.run(data_string)

    # Build our resample calculation
    X = network['network'][0] # 0 to debatch
    Y = network['Y'][0]       # 0 to debatch
    X = tf.reduce_sum(tf.abs(X - Y), axis=1)
    X = X / tf.reduce_sum(X) # Normalise to make sum to 1

    # Loop through the data
    while True:
        try:
            # Get the difference between our labels and our expectations
            result = sess.run([X, network['files']], feed_dict={ network['handle']: data_handle })

            # Our output file
            output_file = result[1][0][3]

            if os.path.exists(output_file):
                with open(output_file, 'rb') as f:
                    print(output_file)
                    # Read and then exp to undo log
                    base = np.exp(np.frombuffer(f.read(), np.float32))
            else:
                # Everything has 0 probability initially
                base = np.zeros_like(result[0])

            # Add on our probabilities to the base, normalise and log
            v = base + result[0] + np.ones_like(result[0]) * (5.0 / len(result[0]))
            v = np.log(v / np.sum(v))

            # Write to file
            print(output_file, len(v) * 4)
            with open(output_file, 'wb') as f:
                f.write(v.tobytes())

        except tf.errors.OutOfRangeError:
            print('Resampling Done')
            break
