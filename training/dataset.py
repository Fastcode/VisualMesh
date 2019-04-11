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
  raise Exception("Please build the tensorflow visual mesh op before running")


class VisualMeshDataset:

  def __init__(self, input_files, classes, geometry, batch_size, shuffle_size, variants):
    self.input_files = input_files
    self.classes = classes
    self.batch_size = batch_size
    self.geometry = tf.constant(geometry['shape'], dtype=tf.string, name='GeometryType')
    self.shuffle_buffer_size = shuffle_size

    self._variants = variants

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
    if 'mesh' in self._variants:
      v = self._variants['mesh']
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

    # Bilinearly interpolate our image based on our floating point pixel coordinate
    y_0 = tf.floor(pixels[:, 0])
    x_0 = tf.floor(pixels[:, 1])
    y_1 = y_0 + 1
    x_1 = x_0 + 1

    # Weights for the x and y axis
    y_w = pixels[:, 0] - y_0
    x_w = pixels[:, 1] - x_0


    # Pixel coordinates to values to weighted values to X
    p_idx = [
      tf.cast(tf.stack([a, b], axis=-1), tf.int32) for a, b in [
        (y_0, x_0),
        (y_0, x_1),
        (y_1, x_0),
        (y_1, x_1),
      ]
    ]
    p_val = [
      tf.image.convert_image_dtype(tf.gather_nd(args['image'], idx), tf.float32) for idx in p_idx
    ]
    p_weighted = [
      tf.multiply(val, tf.expand_dims(w, axis=-1)) for val, w in zip(
        p_val, [
          tf.multiply(1 - y_w, 1 - x_w),
          tf.multiply(1 - y_w, x_w),
          tf.multiply(y_w, 1 - x_w),
          tf.multiply(y_w, x_w),
        ]
      )
    ]
    X = tf.add_n(p_weighted)

    # For the segmentation just use the nearest neighbour
    Y = tf.gather_nd(args['mask'], tf.cast(tf.round(pixels), tf.int32))

    return X, Y, neighbours, pixels

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

  def _apply_variants(self, X):
    # Make the shape of X back into an imageish shape for the functions
    X = tf.expand_dims(X, axis=0)

    # Apply the variants that were listed
    var = self._variants['image']
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

    # Remove the extra dimension we added
    return tf.squeeze(X, axis=0)

  def _reduce_batch(self, state, proto):

    # Load the example from the proto
    example = self._load_example(proto)

    # Project the visual mesh for this example
    X, Y, G, px = self._project_mesh(example)

    # Apply any visual augmentations we may want
    if 'image' in self._variants:
      X = self._apply_variants(X)

    # Expand the classes for this value
    Y, W = self._expand_classes(Y)

    # Add the size of this element on to our n vector
    n = tf.concat([state['n'], tf.expand_dims(tf.shape(Y)[0], axis=0)], axis=0)

    # Concatenate X with the new X, and move the -1 to the end for the null point
    X = tf.concat([state['X'][:-1], X, state['X'][-1:]], axis=0)

    # Concatenate the Y, W, px and raw vectors
    Y = tf.concat([state['Y'], Y], axis=0)
    W = tf.concat([state['W'], W], axis=0)
    px = tf.concat([state['px'], px], axis=0)
    raw = tf.concat([state['raw'], tf.expand_dims(example['raw'], axis=0)], axis=0)

    # Concatenate the graph, and adjust the offsets to be consistent
    # Also update the coordinate of the null point for existing state
    # We can use the shape of Y for this task as it does not have the extra null point on the end so its
    # size will be the index of the null point
    current_n_elems = tf.shape(state['Y'])[0]  # Previous last index
    next_n_elems = tf.shape(Y)[0]  # New last index
    G = tf.concat(
      [
        tf.where(
          state['G'][:-1] == current_n_elems,
          tf.broadcast_to(next_n_elems, shape=tf.shape(state['G'][:-1])),
          state['G'][:-1],
        ),
        G + current_n_elems,
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
          'n': tf.zeros([0], dtype=tf.int32),  # 0 size list of number of points
          'px': tf.zeros([0, 2], dtype=tf.float32),  # 0 * 2 pixel coordinates
          'raw':
            tf.zeros([0], dtype=tf.string)  # List of raw images
        },
        self._reduce_batch
      ),
      num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # Prefetch a few elements
    dataset = dataset.prefetch(10)

    # Prefetch to the GPU
    dataset = dataset.apply(tf.data.experimental.copy_to_device('/device:GPU:0'))
    dataset = dataset.prefetch(5)

    return dataset
