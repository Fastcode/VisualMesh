#!/usr/bin/env python3

# Copyright (C) 2017-2020 Trent Houliston <trent@houliston.me>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import os

import yaml

import tensorflow as tf
import training.testing as testing
import training.training as training
import training.export as export
import training.find_lr as find_lr

if __name__ == "__main__":

    # Parse our command line arguments
    command = argparse.ArgumentParser(description="Utility for training a Visual Mesh network")

    command.add_argument("command", choices=["train", "test", "export", "find_lr"], action="store")
    command.add_argument("network_path", action="store", help="Path to the network folder")
    command.add_argument("-c", "--config", help="Override for the configuration path, default is <network>/config.yaml")

    args = command.parse_args()

    network_path = args.network_path
    config_path = args.config if args.config is not None else os.path.join(network_path, "config.yaml")

    if not os.path.exists(config_path) or not os.path.isfile(config_path):
        print("The configuration file {} does not exist".format(config_path))
        exit(1)

    # Load our yaml file and convert it to an object
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Make all GPUs grow memory as needed
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Run the appropriate action
    if args.command == "train":
        training.train(config, network_path)

    elif args.command == "test":
        testing.test(config, network_path)

    elif args.command == "export":
        export.export(config, network_path)

    elif args.command == "find_lr":
        find_lr.find_lr(config, network_path)
