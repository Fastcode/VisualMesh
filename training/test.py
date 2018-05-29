#!/usr/bin/env python3

import os
import re
import math
import json
import yaml
import numpy as np
import tensorflow as tf

from . import dataset

def test(sess,
         network,
         mesh_type,
         mesh_size,
         model_dir,
         input_path,
         output_path):

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

    # Load our test data
    data_string = dataset.dataset(
        files[round(len(files) * 0.8):],
        mesh_type=mesh_type,
        variants=False,
        repeat=1,
        batch_size=10,
        shuffle=False
    )

    # Get our iterator handle
    data_handle = sess.run(data_string)


    classification = [network['network'][..., 0],
                      network['Y'][..., 0],
                      network['files'][..., 1],
    ]

    results = []
    buckets = 1000

    with open(os.path.join(output_path), 'w') as output_file:

        # Loop through the data
        while True:
            try:

                # Get the difference between our labels and our expectations
                # for tp, tn, fp, fn, f in zip(*sess.run(confusion, feed_dict={ network['handle']: data_handle })):
                for X, Y, f in zip(*sess.run(classification, feed_dict={ network['handle']: data_handle })):

                    meta_file = re.sub(r'(.+)image(\d+)\.jpg', r'\1meta\2.json', f.decode('utf-8'))

                    fno = int(re.match(r'.+image(\d+)\.jpg', f.decode('utf-8')).group(1))

                    with open(meta_file, 'r') as f:
                        meta = json.load(f)

                    distance = np.array(meta['ball']['position'] + [meta['camera']['height']])
                    distance = np.linalg.norm(distance)

                    tp = [0] * (buckets + 1)
                    fp = [0] * (buckets + 1)

                    for x, y in zip(X, Y):
                        b = math.floor(x * buckets)
                        if y == 0:
                            fp[b] += 1
                        else:
                            tp[b] += 1

                    json.dump({
                        'fno': fno,
                        'd': float(distance),
                        'tp': tp,
                        'fp': fp,
                    }, output_file)
                    output_file.write('\n')
                    output_file.flush()

                    print(f'Testing file {fno}')

            except tf.errors.OutOfRangeError:
                print('Testing Done')
                break
