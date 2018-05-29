#!/usr/bin/env python3

import tensorflow as tf
import os
import re

op_file = os.path.join(os.path.dirname(__file__), 'visualmesh_op.so')

if os.path.isfile(op_file):
  VisualMesh = tf.load_op_library(op_file).visual_mesh
else:
  raise Exception("Please build the tensorflow visual mesh op before running training")


class VisualMeshDataset:

  def __init__(self, input_files, classes, geometry, resample_files=None):
    self.input_files = input_files
    self.classes = classes
    self.resample_files = resample_files
    self.geometry = tf.constant(geometry['shape'], dtype=tf.string, name='GeometryType')

    if geometry['shape'] in ['CIRCLE', 'SPHERE']:
      self.geometry_params = tf.constant([geometry['radius'], geometry['intersections'], geometry['max_distance']],
                                         dtype=tf.float32,
                                         name='GeometryParams')

    elif geometry['shape'] in ['CYLINDER']:
      self.geometry_params = tf.constant([
        geometry['height'], geometry['radius'], geometry['intersections'], geometry['max_distance']
      ],
                                         dtype=tf.float32,
                                         name='GeometryParams')
    else:
      raise Exception('Unknown geometry type {}'.format(self.geometry))

  def load_example(self, proto):
    example = tf.parse_single_example(
      proto, {
        'image': tf.FixedLenFeature([], tf.string),
        'mask': tf.FixedLenFeature([], tf.string),
        'lens/projection': tf.FixedLenFeature([], tf.string),
        'lens/focal_length': tf.FixedLenFeature([1], tf.float32),
        'lens/fov': tf.FixedLenFeature([1], tf.float32),
        'mesh/orientation': tf.FixedLenFeature([3, 3], tf.float32),
        'mesh/height': tf.FixedLenFeature([1], tf.float32),
      }
    )

    return {
      'image': tf.image.decode_image(example['image']),
      'mask': tf.image.decode_png(example['mask'], channels=4),
      'projection': example['lens/projection'],
      'focal_length': example['lens/focal_length'],
      'fov': example['lens/fov'],
      'orientation': example['mesh/orientation'],
      'height': example['mesh/height']
    }

  def load_resample(self, proto):
    example = tf.parse_single_example(proto, {'probability': tf.FixedLenFeature([], tf.string)})
    return tf.image.decode_image(example['probability'])

  def project_mesh(self, args):

    pixels, neighbours = VisualMesh(
      tf.shape(args['image']),
      args['projection'],
      args['focal_length'],
      args['fov'],
      args['orientation'],
      args['height'],
      self.geometry,
      self.geometry_params,
      name='ProjectVisualMesh',
    )

    return {
      'pixels': pixels,
      'neighbours': neighbours,
    }
    # TODO consider randomizing the orentation + height by some amount here

    # Return the mesh points from the data and the graph connections
    # return X, Y, G

  def expand_class(self, X, Y, G):

    self.classes

    # Return the expanded classes
    return X, Y, G

  def calculate_weights(self, X, Y, G, R):

    # R if R exists else 1s sizeof Y

    # W = Multiply R by Ys alpha

    # TODO W needs to be calculated such that each class has equal weight, and within those weights the resample + opacity sum to 1

    # Convert Ys into columns
    return X, Y, G, W

  def build(self):
    # Load our dataset
    dataset = tf.data.TFRecordDataset(self.input_files)
    dataset = dataset.map(self.load_example)

    # Try to load the resample dataset if we have one
    resample_dataset = tf.data.TFRecordDataset(self.resample_files).map(
      self.load_resample
    ) if self.resample_files else None

    dataset = dataset.map(self.project_mesh)

    # # Load our resample dataset if it exists
    # if (self.resample_files is not None):
    #   resample_dataset = tf.data.TFRecordDataset(self.input_files)
    #   resample_dataset = resample_dataset.map(self.load_resample)

    # # Project our visual mesh onto the dataset
    # dataset = dataset.map(self.project_mesh)

    # Zip our resample weights into our main dataset
    # dataset = tf.data.Dataset.zip(dataset, resample_dataset)

    return dataset


def main():
  import sys
  ds = VisualMeshDataset(
    sys.argv[1],
    {
      'ball': 255,
      'environment': 0,
    },
    {
      'shape': 'SPHERE',
      'radius': 0.075,
      'max_distance': 15,
      'intersections': 4,
    },
  )

  ds = ds.build().make_one_shot_iterator().get_next()

  with tf.Session() as sess:

    sample = sess.run(ds)
    print(sample)


if __name__ == '__main__':
  main()
