#!/usr/bin/env python3

import tensorflow as tf
import os
import re

def get_files(dir, mesh_size):

    # Load the directory that has our files
    files = sorted([os.path.join(dir, f) for f in os.listdir(dir) if f.startswith('{}mesh'.format(mesh_size))])

    # Get the stencil and image files for our mesh
    files = [(
        f,
        re.sub(r'(.*)\d+mesh(\d+).*', r'\1image\2.png', f),
        re.sub(r'(.*)\d+mesh(\d+).*', r'\1stencil\2.png', f)
    ) for f in files]

    return files

def load_data_files(files, mesh_type='VISUALMESH'):

    # Load our image and stencil
    image = tf.image.decode_png(tf.read_file(files[1], name='ReadImage'), 3, name='DecodeImage')
    image = tf.image.convert_image_dtype(image, tf.float32, name='CastImage')
    stencil = tf.image.decode_png(tf.read_file(files[2], name='ReadClasses'), 1, name='DecodeClasses')

    # Read our mesh data as int32s
    mesh_data = tf.decode_raw(tf.read_file(files[0], name='ReadMesh'), tf.int32, name='DecodeMesh')

    # Work out how many mesh elements we have
    # tf.size(mesh_data) == 2(n-1) + 7(n)
    n = (tf.size(mesh_data) + 1) // 9

    # Grab our pixel coordinates and neighbourhood
    if mesh_type == 'VISUALMESH':
        pixel_coordinates = tf.reshape(mesh_data[:(n - 1) * 2], shape=[-1, 2], name='GetPixelCoordinates')

        # Get our neighbourhood indices
        neighbourhood = tf.reshape(mesh_data[(n - 1) * 2:-1], shape=[-1, 7], name='GetNeighbourhoodIndices')

    elif mesh_type in ['HEXEQUAL', 'HEXDENSE']:

        if mesh_type == 'HEXEQUAL':
            img_size = tf.shape(stencil)
            c1 = tf.sqrt(n);
            c2 = tf.sqrt(img_size[0] / img_size[1]);
            n_cols   = c1 * c2;
            n_rows   = c1 / c2;
        elif mesh_type == 'HEXDENSE':
            n_cols = tf.min(img_size[0], img_size[0] / pixel_size);
            n_rows = tf.min(img_size[1], img_size[1] / pixel_size);

        # TODO generate pixel coordinates and neighbourhood graph
        tf.linspace(begin, end, n_cols)
        tf.linspace(begin, end, n_rows)

    # Read our pixel size as a float
    pixel_size = tf.bitcast(mesh_data[-1], tf.float32)

    return (
        image,
        stencil,
        neighbourhood,
        pixel_coordinates
    )

def select_points(x, y, g):

    x = tf.squeeze(x, name='SqueezeImageDimensions')

    # Add our offscreen points to x and y
    x = tf.pad(x, [[1, 0], [0, 0]], name='PadOffscreenImage')
    y = tf.pad(y, [[1, 0]], name='PadOffscreenClasses')

    # If you get more classes, put them here
    classes = [
        255,    # Ball
        0       # None
    ]

    # Gather the indices to the ball points
    # We select each ball point twice to let more other points come in
    ball_points = tf.tile(tf.squeeze(tf.where(tf.equal(y, classes[0])), [1]), multiples=[2], name='DuplicateBallPoints')

    # Get non ball points of equal number to ball points
    other_points = tf.squeeze(tf.where(tf.equal(y, classes[1])))
    other_points = tf.random_shuffle(other_points)[:tf.size(ball_points)]

    # List of our indices
    yi = tf.cast(tf.concat([ball_points, other_points], axis=0), tf.int32)

    # Convert our single value classes into multiple outputs
    ys = tf.stack([tf.cast(tf.equal(y, v), tf.float32) for v in classes], axis=1)

    # Gather the relevant ones
    ys = tf.gather(ys, yi)

    return x, g, ys, yi


def dataset(files, variants=True, repeat=1, batch_size=10, shuffle=True):

    # Make a constant tensor to hold them
    dataset = tf.constant(files)

    # Create our dataset from our list of files
    dataset = tf.data.Dataset.from_tensor_slices(dataset)

    # Repeat for epochs
    dataset = dataset.repeat(repeat)

    # Load our images
    dataset = dataset.map(load_data_files, 8)

    # Extract our relevant points from our image
    dataset = dataset.map(lambda x, y, g, px: (tf.expand_dims(tf.gather_nd(x, px), 0), tf.squeeze(tf.gather_nd(y, px)), g), 8)

    # Filter out images that are all empty (no objects of interest)
    dataset = dataset.filter(lambda x, y, g: tf.count_nonzero(y) > 0)

    if variants:
        # Repeat number of variant times
        dataset = dataset.repeat(10)

        # Apply random brightness
        dataset = dataset.map(lambda x, y, g: (tf.image.random_brightness(x, 0.5), y, g), 8)

        # Apply random contrast
        dataset = dataset.map(lambda x, y, g: (tf.image.random_contrast(x, 0.5, 1.5), y, g), 8)

        # Apply random saturation
        dataset = dataset.map(lambda x, y, g: (tf.image.random_saturation(x, 0.5, 1.5), y, g), 8)

        # Apply random hue change
        dataset = dataset.map(lambda x, y, g: (tf.image.random_hue(x, 0.2), y, g), 8)

    # Select our points
    dataset = dataset.map(select_points, 8)

    # Shuffle points
    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 3)

    # Pad and batch
    dataset = dataset.padded_batch(batch_size, ([-1, -1], [-1, 7], [-1, -1], [-1]))

    return dataset.make_one_shot_iterator().string_handle()
