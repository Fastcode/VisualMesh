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
import training.export as export
import training.find_lr as find_lr
import training.testing as testing
import training.training as training
import training.unexport as unexport

if __name__ == "__main__":

    # Add command parsers
    command = argparse.ArgumentParser(description="Utility for training a Visual Mesh network")
    subcommands = command.add_subparsers(
        dest="command", help="The command to run from the script. See each help for more information."
    )

    # List of commands
    train_command = subcommands.add_parser("train")
    test_command = subcommands.add_parser("test")
    export_command = subcommands.add_parser("export")
    export_command = subcommands.add_parser("unexport")
    find_lr_command = subcommands.add_parser("find_lr")

    # Add common arguments
    for c in [train_command, test_command, export_command, find_lr_command]:
        c.add_argument("network_path", action="store", help="Path to the network folder")
        c.add_argument(
            "-c", "--config", help="Override for the configuration path, default is <output_path/config.yaml>"
        )

    # Find LR command
    find_lr_command.add_argument("--min_lr", type=int, default=1e-6, help="The minimum learning rate to search from")
    find_lr_command.add_argument("--max_lr", type=int, default=1e2, help="The maximum learning rate to search to")
    find_lr_command.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="The number of steps to take while searching from the minimum to maximum learning rate",
    )
    find_lr_command.add_argument(
        "--window_size", type=int, default=250, help="The size of the averaging window used for smoothing the loss"
    )

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
    gpus = tf.config.list_physical_devices("GPU")
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

    elif args.command == "unexport":
        unexport.unexport(config, network_path)

    elif args.command == "find_lr":
        find_lr.find_lr(config, network_path, args.min_lr, args.max_lr, args.steps, args.window_size)
