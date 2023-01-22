#!/usr/bin/env python3

import argparse
import os
import re
import sys
from glob import glob

import numpy as np
import yaml
from tqdm import tqdm

import tensorflow as tf


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def make_tfrecord(output_file, input_files):

    with tf.io.TFRecordWriter(output_file) as writer:

        for image_file, mask_file, lens_file in tqdm(
            input_files,
            desc="Creating {}".format(os.path.basename(output_file)),
            leave=True,
            unit="files",
            dynamic_ncols=True,
        ):

            with open(lens_file, "r") as f:
                lens = yaml.safe_load(f)
            with open(image_file, "rb") as f:
                image = f.read()
            with open(mask_file, "rb") as f:
                mask = f.read()

            # Create the record
            writer.write(
                tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "image": bytes_feature(image),
                            "mask": bytes_feature(mask),
                            "lens/projection": bytes_feature(lens["projection"].encode("utf-8")),
                            "lens/fov": float_feature(lens["fov"]),
                            "lens/focal_length": float_feature(lens["focal_length"]),
                            "lens/centre": float_list_feature(lens["centre"]),
                            "lens/k": float_list_feature(lens["k"]),
                            "Hoc": float_list_feature(np.array(lens["Hoc"]).flatten().tolist()),
                        }
                    )
                ).SerializeToString()
            )


if __name__ == "__main__":

    # Parse our command line arguments
    command = argparse.ArgumentParser(description="Utility for training a Visual Mesh network")
    command.add_argument("input_path", action="store", help="Path to the input files")
    command.add_argument("output_path", action="store", help="Path to place the output tfrecord files")
    args = command.parse_args()
    input_path = args.input_path
    output_path = args.output_path

    image_files = glob(os.path.join(input_path, "image*.jpg"))
    mask_files = glob(os.path.join(input_path, "mask*.png"))
    lens_files = glob(os.path.join(input_path, "lens*.yaml"))

    # Extract which numbers are in each of the folders
    image_re = re.compile(r"image([^.]+)\.jpg$")
    mask_re = re.compile(r"mask([^.]+)\.png$")
    lens_re = re.compile(r"lens([^.]+)\.yaml$")
    image_nums = set([image_re.search(os.path.basename(f)).group(1) for f in image_files])
    mask_nums = set([mask_re.search(os.path.basename(f)).group(1) for f in mask_files])
    lens_nums = set([lens_re.search(os.path.basename(f)).group(1) for f in lens_files])
    common_nums = image_nums & mask_nums & lens_nums

    files = [
        (
            os.path.join(input_path, "image{}.jpg".format(n)),
            os.path.join(input_path, "mask{}.png".format(n)),
            os.path.join(input_path, "lens{}.yaml".format(n)),
        )
        for n in common_nums
    ]

    nf = len(files)

    training = 0.45
    validation = 0.10

    test = (round(nf * (training + validation)), nf)
    validation = (round(nf * training), round(nf * (training + validation)))
    training = (0, round(nf * training))

    # Create the output folder
    os.makedirs(output_path, exist_ok=True)

    # Create the three datasets
    make_tfrecord(os.path.join(output_path, "training.tfrecord"), files[training[0] : training[1]])
    make_tfrecord(os.path.join(output_path, "validation.tfrecord"), files[validation[0] : validation[1]])
    make_tfrecord(os.path.join(output_path, "testing.tfrecord"), files[test[0] : test[1]])
