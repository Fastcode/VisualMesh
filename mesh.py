#!/usr/bin/env python3

import os
import argparse
import yaml

import tensorflow as tf
import training.training as training
import training.test as test

if __name__ == "__main__":

  # Parse our command line arguments
  command = argparse.ArgumentParser(description='Utility for training a Visual Mesh network')

  command.add_argument('command', choices=['train', 'test'], action='store')
  command.add_argument('config', action='store', help='Path to the configuration file for training')
  command.add_argument('output_path', nargs='?', action='store', help='Output directory to store the logs and models')

  args = command.parse_args()

  # Load our yaml file
  with open(args.config) as f:
    config = yaml.load(f)

  output_path = 'output' if args.output_path is None else args.output_path

  # Run the appropriate action
  if args.command == 'train':
    training.train(config, output_path)

  elif args.command == 'test':
    test.test(config, output_path)
