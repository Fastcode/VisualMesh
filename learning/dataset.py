#!/usr/bin/env python3

import tensorflow as tf
import os
import re


class VisualMeshDataset:

  def __init__(self, input_files, classes, resample_files=None, mesh_intersections=4):
    self.input_files = input_files
    self.classes = classes
    self.resample_files = resample_files
    self.mesh_intersections = mesh_intersections

  def load_example(self, proto):
    example = tf.parse_single_example(
      proto, {
        'image': tf.FixedLenFeature([], tf.string),
        'stencil': tf.FixedLenFeature([], tf.string),
        'lens/projection': tf.FixedLenFeature([], tf.string),
        'lens/focal_length': tf.FixedLenFeature([1], tf.float32),
        'lens/fov': tf.FixedLenFeature([1], tf.float32),
        'orientation': tf.FixedLenFeature([9], tf.float32),
        'height': tf.FixedLenFeature([2], tf.float32),
      })

    return (tf.image.decode_image(example['image']), tf.image.decode_png(example['stencil'],
                                                                         channels=4), example['lens/projection'],
            example['lens/focal_length'], example['lens/fov'], example['orientation'], example['height'])

  def load_resample(self, proto):
    example = tf.parse_single_example(
      proto, {
        'image': tf.FixedLenFeature([], tf.string),
        'stencil': tf.FixedLenFeature([], tf.string),
        'lens/projection': tf.FixedLenFeature([], tf.string),
        'lens/focal_length': tf.FixedLenFeature([1], tf.float32),
        'lens/fov': tf.FixedLenFeature([1], tf.float32),
        'orientation': tf.FixedLenFeature([9], tf.float32),
        'height': tf.FixedLenFeature([2], tf.float32),
      })
    return example['sample']

  def project_mesh(self, image, stencil, lens_projection, lens_focal_length, lens_fov, orientation, height):

    # TODO consider randomizing the orentation + height by some amount here

    # Return the mesh points from the data and the graph connections
    return X, Y, G

  def expand_class(self, X, Y, G):

    self.classes

    # Return the expanded classes
    return X, Y, G

  def calculate_weights(self, X, Y, G, R):

    # R if R exists else 1s sizeof Y

    # W = Multiply R by Ys alpha

    # TODO W needs to be calulcated such that each class has equal weight, and within those weights the resample + opacity sum to 1

    # Convert Ys into columns
    return X, Y, G, W

  def build(self):
    # Load our dataset
    dataset = tf.data.TFRecordDataset(self.input_files)
    dataset = dataset.map(self.load_example)

    # Load our resample dataset if it exists
    if (self.resample_files is not None):
      resample_dataset = tf.data.TFRecordDataset(self.input_files)
      resample_dataset = resample_dataset.map(self.load_resample)

    # Project our visual mesh onto the dataset
    dataset = dataset.map(project_mesh)

    # Zip our resample weights into our main dataset
    dataset = tf.data.Dataset.zip(dataset, resample_dataset)

    return dataset
