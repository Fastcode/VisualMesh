#!/usr/bin/env python3

import os
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
        variants=False,
        repeat=1,
        batch_size=1,
        shuffle=False
    )

    # Get our iterator handle
    data_handle = sess.run(data_string)

    # Loop through the data
    while True:
        try:
            # Run our training step
            result = sess.run([network['network'], network['files']], feed_dict={
                network['handle']: data_handle
            })

            import pdb
            pdb.set_trace()

        except tf.errors.OutOfRangeError:
            print('Resampling Done')
            break
