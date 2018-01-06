#!/usr/bin/env python3

import tensorflow as tf
import os
import re
import yaml


def show(sess, network, model_dir):

    # Initialise global variables
    sess.run(tf.global_variables_initializer())

    save_vars = {v.name: v for v in tf.trainable_variables()}
    saver = tf.train.Saver(save_vars)

    # Get our model directory and load it if it exists
    model_path = os.path.join(model_dir, 'model.ckpt')
    if os.path.isfile(os.path.join(model_dir, 'checkpoint')):
        print('Loading model {}'.format(model_path))
        saver.restore(sess, model_path)
    else:
        print('Model not found')
        exit(1)

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
    print(yaml.dump(output, width=120))
