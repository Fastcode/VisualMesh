#!/usr/bin/env python3

import os
import argparse
import yaml

# import tensorflow as tf
# import learning.network as network
# import learning.training as training
# import learning.resample as resample
# import learning.test as test

if __name__ == "__main__":

  # Parse our command line arguments
  command = argparse.ArgumentParser(description='Utility for training a Visual Mesh network')

  command.add_argument('command', choices=['train', 'test', 'resample'], action='store')
  command.add_argument('-g', '--gpu', action='store', default=0, help='The index of the GPU to use when training')
  command.add_argument('config', action='store', help='Path to the configuration file for training')

  args = command.parse_args()

  # Load our yaml file
  with open(args.config) as f:
    config = yaml.load(f)

  # Tensorflow configuration
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  config.graph_options.build_cost_model = 1
  config.gpu_options.allow_growth = True

  with tf.Session(config=config) as sess:

    # Select our device to run operations on
    with tf.device('/device:GPU:{}'.format(args.gpu)):

      # Build our network
      net = network.build(config['structure'])

      # Run the appropriate action
      if args.command == 'train':
        training.train(sess, net)

      elif args.command == 'resample':
        resample.resample(sess, net)

      elif args.command == 'test':
        test.test(sess, net)
