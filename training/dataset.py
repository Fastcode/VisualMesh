#!/usr/bin/env python3

import tensorflow as tf
import os
import re
import multiprocessing

op_file = os.path.join(os.path.dirname(__file__), 'visualmesh_op.so')

if os.path.isfile(op_file):
  VisualMesh = tf.load_op_library(op_file).visual_mesh
else:
  raise Exception("Please build the tensorflow visual mesh op before running training")


class VisualMeshDataset:

  def __init__(self, input_files, classes, geometry, batch_size, variants, resample_files=None):
    self.input_files = input_files
    self.classes = classes
    self.batch_size = batch_size
    self.geometry = tf.constant(geometry['shape'], dtype=tf.string, name='GeometryType')
    self.resample_files = resample_files
    self.shuffle_buffer_size = 10

    self.variants = variants

    # Convert our geometry into a set of numbers
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
      'image': tf.image.decode_image(example['image'], channels=3),
      'mask': tf.image.decode_png(example['mask'], channels=4),
      'projection': example['lens/projection'],
      'focal_length': example['lens/focal_length'],
      'fov': example['lens/fov'],
      'orientation': example['mesh/orientation'],
      'height': example['mesh/height']
    }

  def load_resample(self, proto):
    example = tf.parse_single_example(proto, {'probability': tf.FixedLenFeature([], tf.string)})
    return tf.image.decode_png(example['probability'], channels=4)

  def merge_resample(self, args, resample):

    # Add the resample results into the dictionary
    return {
      'resample':
        tf.cond(
          tf.reduce_all(tf.equal(tf.shape(resample), tf.shape(args['mask']))),
          lambda: resample,
          lambda: tf.ones_like(args['mask']) * 255,
        ),
      **args,
    }

  def project_mesh(self, args):

    # Adjust our height by a random amount
    height = args['height'] + tf.random_normal(
      [1],
      self.variants['mesh']['height']['mean'],
      self.variants['mesh']['height']['stddev'],
    )

    # TODO create a random rotation for orientation

    # Run the visual mesh to get our values
    pixels, neighbours = VisualMesh(
      tf.shape(args['image']),
      args['projection'],
      args['focal_length'],
      args['fov'],
      args['orientation'],
      height,
      self.geometry,
      self.geometry_params,
      name='ProjectVisualMesh',
    )

    # Round to integer pixels
    # TODO one day someone could do linear interpolation here, like what happens in the OpenCL version
    pixels = tf.cast(tf.round(pixels), dtype=tf.int32)

    # Select the points in the network and discard the old dictionary data
    # We pad one extra point at the end for the offscreen point
    return {
      'X': tf.pad(tf.gather_nd(args['image'], pixels), [[0, 1], [0, 0]]),
      'Y': tf.pad(tf.gather_nd(args['mask'], pixels), [[0, 1], [0, 0]]),
      'S': tf.pad(tf.gather_nd(args['resample'], pixels), [[0, 1], [0, 0]]),
      'G': neighbours,
    }

  def flatten_batch(self, args):

    # This adds an offset to the graph indices so they will be correct once flattened
    G = args['G'] + tf.cumsum(args['n'], exclusive=True)[:, tf.newaxis, tf.newaxis]

    # Find the indices of valid points from the mask
    idx = tf.where(tf.squeeze(args['m'], axis=-1))

    # Use this to lookup each of the vectors
    X = tf.gather_nd(args['X'], idx)
    Y = tf.gather_nd(args['Y'], idx)
    S = tf.gather_nd(args['S'], idx)
    G = tf.gather_nd(G, idx)

    return {
      'X': X,
      'Y': Y,
      'S': S,
      'G': G,
    }

  def expand_classes(self, args):

    # Apply the alpha values from the classes to our weights
    S = tf.multiply(
      tf.image.convert_image_dtype(args['S'][:, 1], tf.float32),
      tf.image.convert_image_dtype(args['Y'][:, 3], tf.float32),
    )

    # Expand the classes from colours into individual columns
    Y = args['Y']
    cs = []
    for name, value in self.classes:
      cs.append(
        tf.where(
          tf.reduce_all(tf.equal(Y[:, :3], [value]), axis=-1),
          tf.fill([tf.shape(Y)[0]], 1.0),
          tf.fill([tf.shape(Y)[0]], 0.0),
        )
      )
    Y = tf.stack(cs, axis=-1)

    return {
      'X': args['X'],
      'Y': Y,
      'S': S,
      'G': args['G'],
    }

  def calculate_weights(self, args):

    # Work out how to arrange the weights of points such that
    W = args['S']
    scatters = []
    for i in range(len(self.classes)):
      idx = tf.where(args['Y'][:, i])
      points = tf.gather_nd(W, idx)
      weights = tf.multiply(tf.nn.softmax(points), tf.cast(tf.shape(W)[0], tf.float32) / len(self.classes))
      scatters.append(tf.scatter_nd(idx, weights, tf.shape(W, out_type=tf.int64)))
    W = tf.add_n(scatters)

    return {
      'X': args['X'],
      'Y': args['Y'],
      'W': W,
      'G': args['G'],
    }

  def build(self):
    # Load our dataset
    dataset = tf.data.TFRecordDataset(self.input_files)
    dataset = dataset.map(self.load_example, num_parallel_calls=multiprocessing.cpu_count())

    # Try to load the resample dataset if we have one
    resample_dataset = tf.data.TFRecordDataset(self.resample_files).map(
      self.load_resample,
      num_parallel_calls=multiprocessing.cpu_count(),
    ) if self.resample_files else tf.data.Dataset.from_tensors(tf.constant([], dtype=tf.uint8)).repeat()

    # Merge in the resample dataset
    dataset = dataset.zip((dataset, resample_dataset))
    dataset = dataset.map(self.merge_resample, num_parallel_calls=multiprocessing.cpu_count())

    # Before we get to the point of making variants etc, shuffle here
    dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)

    # Project the visual mesh and select the appropriate pixels
    dataset = dataset.map(self.project_mesh, num_parallel_calls=multiprocessing.cpu_count())

    # Batch the visual mesh by concatenating meshes and graphs and updating the graph coordinates to match
    dataset = dataset.map(
      lambda args: {
        **args,
        'n': tf.shape(args['X'])[0],
        'm': tf.fill((tf.shape(args['X'])[0], 1), True)}
    )
    dataset = dataset.prefetch(self.batch_size * 2)
    dataset = dataset.padded_batch(
      batch_size=self.batch_size,
      padded_shapes={
        'X': (None, 3),
        'Y': (None, 4),
        'S': (None, 4),
        'G': (None, 7),
        'n': (),
        'm': (None, 1)
      },
    )
    dataset = dataset.map(self.flatten_batch, num_parallel_calls=multiprocessing.cpu_count())

    # Expand the classes
    dataset = dataset.map(self.expand_classes, num_parallel_calls=multiprocessing.cpu_count())

    # Calculate the weights to balance classes and resampling
    dataset = dataset.map(self.calculate_weights, num_parallel_calls=multiprocessing.cpu_count())

    # TODO spread the classes
    # TODO calculate the weights
    # TODO apply the hue etc variants

    return dataset


def main():
  import sys
  ds = VisualMeshDataset(
    input_files=sys.argv[1],
    classes=[
      ['ball', [255, 255, 255]],
      ['environment', [0, 0, 0]],
    ],
    batch_size=50,
    variants={
      'mesh': {
        'height': {
          'mean': 0,
          'stddev': 0.1,
        },
        'rotation': {
          'mean': 0,
          'stddev': 0.1,
        },
      },
    },
    geometry={
      'shape': 'SPHERE',
      'radius': 0.075,
      'max_distance': 15,
      'intersections': 4,
    },
  )

  ds = ds.build().make_one_shot_iterator().get_next()

  with tf.Session() as sess:

    import time
    while True:
      t1 = time.clock()
      sample = sess.run(ds)
      t2 = time.clock()
      print(t2 - t1)

      import pdb
      pdb.set_trace()


if __name__ == '__main__':
  main()
