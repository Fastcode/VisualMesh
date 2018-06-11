#!/usr/bin/env python3

from tqdm import tqdm
import sys
import os
import math
import tensorflow as tf
import json
from PIL import Image
from glob import glob


def float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def make_tfrecord(output_file, input_files):

  writer = tf.python_io.TFRecordWriter(output_file)

  for image_file, mask_file, meta_file in tqdm(input_files):

    with open(meta_file, 'r') as f:
      meta = json.load(f)

    with open(image_file, 'rb') as f:
      image = f.read()
    with open(mask_file, 'rb') as f:
      stencil = f.read()

    im = Image.open(image_file)
    width, height = im.size

    # Convert from blender to visual mesh coordinates
    rot = meta['rotation']
    Roc = [-rot[0][2], -rot[0][0], rot[0][1],
           -rot[1][2], -rot[1][0], rot[1][1],
           -rot[2][2], -rot[2][0], rot[2][1],
    ] # yapf: disable
    height = meta['height']

    projection = meta['lens']['type']
    fov = meta['lens']['fov']
    focal_length = meta['lens']['focal_length']

    features = {
      'image': _bytes_feature(image),
      'mask': _bytes_feature(stencil),
      'lens/projection': _bytes_feature(projection.encode('utf-8')),
      'lens/fov': _float_featur(fov),
      'lens/focal_length': _float_feature(focal_length),
      'mesh/orientation': _float_list_feature(Roc),
      'mesh/height': _float_feature(height),
    }

    example = tf.train.Example(features=tf.train.Features(feature=features))

    writer.write(example.SerializeToString())

  writer.close()


if __name__ == '__main__':

  image_path = sys.argv[1]
  mask_path = sys.argv[2]
  meta_path = sys.argv[3]

  files = sorted(glob(os.path.join(path, 'meta*.json')))
  nf = len(files)

  training = 0.45
  validation = 0.10

  testing = (round(nf * (training + validation)), nf)
  validation = (round(nf * training), round(nf * (training + validation)))
  training = (0, round(nf * training))

  make_tfrecord('training.tfrecord', files[training[0]:training[1]])
  make_tfrecord('validation.tfrecord', files[validation[0]:validation[1]])
  make_tfrecord('testing.tfrecord', files[testing[0]:testing[1]])

  output_file = 'test.tfrecord'
