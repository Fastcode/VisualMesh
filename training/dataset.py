#!/usr/bin/env python3

import tensorflow as tf
import os
import re
import math

# Load the visual mesh op
op_file = os.path.join(os.path.dirname(__file__), 'visualmesh_op.so')
if os.path.isfile(op_file):
  VisualMesh = tf.load_op_library(op_file).visual_mesh
else:
  raise Exception("Please build the tensorflow visual mesh op before running training")


class VisualMeshDataset:

  def __init__(self, input_files, classes, geometry, batch_size, shuffle_size, variants):
    self.input_files = input_files
    self.classes = classes
    self.batch_size = batch_size
    self.geometry = tf.constant(geometry['shape'], dtype=tf.string, name='GeometryType')
    self.shuffle_buffer_size = shuffle_size

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

  def _load_example(self, proto):
    example = tf.parse_single_example(
      proto, {
        'image': tf.FixedLenFeature([], tf.string),
        'mask': tf.FixedLenFeature([], tf.string),
        'lens/projection': tf.FixedLenFeature([], tf.string),
        'lens/focal_length': tf.FixedLenFeature([1], tf.float32),
        'lens/fov': tf.FixedLenFeature([1], tf.float32),
        'lens/centre': tf.FixedLenFeature([2], tf.float32),
        'mesh/orientation': tf.FixedLenFeature([3, 3], tf.float32),
        'mesh/height': tf.FixedLenFeature([1], tf.float32),
      }
    )

    return {
      'image': tf.image.decode_image(example['image'], channels=3),
      'mask': tf.image.decode_png(example['mask'], channels=4),
      'projection': example['lens/projection'],
      'focal_length': example['lens/focal_length'],
      'lens_centre': example['lens/centre'],
      'fov': example['lens/fov'],
      'orientation': example['mesh/orientation'],
      'height': example['mesh/height'],
      'raw': example['image'],
    }

  def _project_mesh(self, args):

    height = args['height']
    orientation = args['orientation']

    # Adjust our height and orientation
    if 'mesh' in self.variants:
      v = self.variants['mesh']
      if 'height' in v:
        height = height + tf.truncated_normal(
          shape=(),
          mean=v['height']['mean'],
          stddev=v['height']['stddev'],
        )
      if 'rotation' in v:
        # Make 3 random euler angles
        rotation = tf.truncated_normal(
          shape=[3],
          mean=v['rotation']['mean'],
          stddev=v['rotation']['stddev'],
        )
        # Cos and sin for everyone!
        ca = tf.cos(rotation[0])
        sa = tf.sin(rotation[0])
        cb = tf.cos(rotation[1])
        sb = tf.sin(rotation[0])
        cc = tf.cos(rotation[2])
        sc = tf.sin(rotation[0])

        # Convert these into a rotation matrix
        rot = [cc*ca, -cc*sa*cb + sc*sb, cc*sa*sb + sc*cb,
                  sa,             ca*cb,         -ca * sb,
              -sc*ca,  sc*sa*cb + cc*sb, -sc*sa*sb + cc*cb]  # yapf: disable
        rot = tf.reshape(tf.stack(rot), [3, 3])

        # Apply the rotation
        orientation = tf.matmul(rot, orientation)

    # Run the visual mesh to get our values
    pixels, neighbours = VisualMesh(
      tf.shape(args['image']),
      args['projection'],
      args['focal_length'],
      args['fov'],
      args['lens_centre'],
      orientation,
      height,
      self.geometry,
      self.geometry_params,
      name='ProjectVisualMesh',
    )

    # Round to integer pixels
    # TODO one day someone could do linear interpolation here, like what happens in the OpenCL version
    int_pixels = tf.cast(tf.round(pixels), dtype=tf.int32)

    # Select the points in the network and discard the old dictionary data
    # We pad one extra point at the end for the offscreen point
    return tf.gather_nd(args['image'], int_pixels), tf.gather_nd(args['mask'], int_pixels), neighbours, pixels

  def _expand_classes(self, Y):

    # Expand the classes from colours into individual columns
    W = tf.image.convert_image_dtype(Y[:, 3], tf.float32)  # Alpha channel
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

    return Y, W

  def apply_variants(self, args):
    # Cast the image to a floating point value and make it into an image shape
    X = tf.expand_dims(tf.image.convert_image_dtype(args['X'], tf.float32), 0)

    # Apply the variants that were listed
    var = self.variants['image']
    if 'brightness' in var and var['brightness']['stddev'] > 0:
      X = tf.image.adjust_brightness(
        X, tf.truncated_normal(
          shape=(),
          mean=var['brightness']['mean'],
          stddev=var['brightness']['stddev'],
        )
      )
    if 'contrast' in var and var['contrast']['stddev'] > 0:
      X = tf.image.adjust_contrast(
        X, tf.truncated_normal(
          shape=(),
          mean=var['contrast']['mean'],
          stddev=var['contrast']['stddev'],
        )
      )
    if 'hue' in var and var['hue']['stddev'] > 0:
      X = tf.image.adjust_hue(X, tf.truncated_normal(
        shape=(),
        mean=var['hue']['mean'],
        stddev=var['hue']['stddev'],
      ))
    if 'saturation' in var and var['saturation']['stddev'] > 0:
      X = tf.image.adjust_saturation(
        X, tf.truncated_normal(
          shape=(),
          mean=var['saturation']['mean'],
          stddev=var['saturation']['stddev'],
        )
      )
    if 'gamma' in var and var['gamma']['stddev'] > 0:
      X = tf.image.adjust_gamma(
        X, tf.truncated_normal(
          shape=(),
          mean=var['gamma']['mean'],
          stddev=var['gamma']['stddev'],
        )
      )

    return {**args, 'X': tf.squeeze(tf.image.convert_image_dtype(X, tf.uint8), 0)}

  def _reduce_batch(self, state, proto):

    # Load the example from the proto
    example = self._load_example(proto)

    # Project the visual mesh for this example
    X, Y, G, px = self._project_mesh(example)

    # Expand the classes for this value
    Y, W = self._expand_classes(Y)

    # Add the size of this element on to our n vector
    n = tf.concat([state['n'], tf.expand_dims(tf.shape(X)[0], axis=0)], axis=0)

    # Concatenate X with the new X, and move the -1 to the end for the null point
    X = tf.image.convert_image_dtype(X, dtype=tf.float32)
    X = tf.concat([state['X'][:-1], X, state['X'][-1:]], axis=0)

    # Concatenate the Y, W, px and raw vectors
    Y = tf.concat([state['Y'], Y], axis=0)
    W = tf.concat([state['W'], W], axis=0)
    px = tf.concat([state['px'], px], axis=0)
    raw = tf.concat([state['raw'], tf.expand_dims(example['raw'], axis=0)], axis=0)

    # Concatenate the graph, and adjust the offsets to be consistent
    # Also update the coordinate of the null point for existing state
    n_elems = tf.shape(state['Y'])[0]
    G = tf.concat(
      [
        tf.where(
          state['G'][:-1] == n_elems,
          tf.broadcast_to(n_elems + n[-1], shape=tf.shape(state['G'][:-1])),
          state['G'][:-1],
        ),
        G + n_elems,
      ],
      axis=0,
    )

    # Return the results
    return {'X': X, 'Y': Y, 'W': W, 'n': n, 'G': G, 'px': px, 'raw': raw}

  def build(self):

    # Load our dataset records
    dataset = tf.data.TFRecordDataset(self.input_files, buffer_size=2**30)

    # Window the dataset by our batch size
    dataset = dataset.window(self.batch_size)

    # Apply our reduction function to project/squash our dataset into a batch
    dataset = dataset.map(
      lambda ds: ds.reduce(
        {
          'X': tf.ones([1, 3], dtype=tf.float32) * -1,  # -1 for null point that will always be at the end
          'Y': tf.zeros([0, len(self.classes)], dtype=tf.float32),  # 0 * num classes
          'W': tf.zeros([0], dtype=tf.float32),  # 0 length Weights
          'G': tf.zeros([0, 7], dtype=tf.int32),  # 0 size Graph Degree
          'n': tf.zeros([1], dtype=tf.int32),  # A single element for the start of the first element
          'px': tf.zeros([0, 2], dtype=tf.float32),  # 0 * 2 pixel coordinates
          'raw':
            tf.zeros([0], dtype=tf.string)  # List of raw images
        },
        self._reduce_batch
      ),
      num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    dataset = dataset.prefetch(10)

    # And prefetch
    dataset = dataset.apply(tf.data.experimental.copy_to_device('/device:GPU:0'))
    dataset = dataset.prefetch(5)

    return dataset
