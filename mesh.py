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

import training.testing as testing
import training.training as training


if __name__ == "__main__":

    # Parse our command line arguments
    command = argparse.ArgumentParser(description="Utility for training a Visual Mesh network")

    command.add_argument("command", choices=["train", "test"], action="store")
    command.add_argument("config", action="store", help="Path to the configuration file for training")
    command.add_argument(
        "output_path", nargs="?", action="store", help="Output directory to store the logs and models",
    )

    args = command.parse_args()

    # Load our yaml file and convert it to an object
    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_path = "output" if args.output_path is None else args.output_path

    # Run the appropriate action
    if args.command == "train":
        training.train(config, output_path)

    elif args.command == "test":
        testing.test(config, output_path)
