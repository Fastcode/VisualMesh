#!/usr/bin/env python3

import os

import tensorflow as tf
import learning.network as network
import learning.training as training
import learning.resample as resample

# Tensorflow configuration
config = tf.ConfigProto()
config.allow_soft_placement = True
config.graph_options.build_cost_model = 1
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
config.gpu_options.force_gpu_compatible = True

with tf.Session(config=config) as sess:


    # Each number is output neurons for that layer, each list in the list is a convolution
    groups = [[5], [5], [5], [5, 2]]
    dropout = 1.0
    regularisation = 0.00
    gpu = 2

    # The name for this network
    network_name = '_'.join(['-'.join([str(l) for l in g]) for g in groups])
    network_name = 'n{}d{:.2f}r{:.3f}'.format(network_name, dropout, regularisation)

    # Input paths
    paths = {
        'trees': os.path.join('training', 'resample'),
        'validation': os.path.join('training', 'validation'),
        'logs': os.path.join('training', 'output', 'logs', network_name),
        'output_validation': os.path.join('training', 'output', 'validation', network_name),
        'output_trees': os.path.join('training', 'rstrees', network_name)
    }

    for k, p in paths.items():
        os.makedirs(p, exist_ok=True)

    # Select our device
    with tf.device('/device:GPU:{}'.format(gpu)):

        # Get our network
        net = network.build(groups)

        # Train our network
        training.train(sess, net, paths, dropout=dropout, regularisation=regularisation)
