#!/usr/bin/env python3

import tensorflow as tf
import os
import re
from PIL import Image, ImageDraw

def get_files(dir, mesh_size):

    # Load the directory that has our files
    files = sorted([os.path.join(dir, f) for f in os.listdir(dir) if f.startswith('{}mesh'.format(mesh_size))])

    # Get the stencil and image files for our mesh
    files = [(
        f,
        re.sub(r'(.*)\d+mesh(\d+).*', r'\1image\2.jpg', f),
        re.sub(r'(.*)\d+mesh(\d+).*', r'\1stencil\2.png', f)
    ) for f in files]

    return files

def draw_image(file, px, neighbourhood):

    file = file.decode('utf-8')

    # Load our input image
    img = Image.open(file)
    d = ImageDraw.Draw(img)

    # Go through our drawing coordinates
    for i in range(len(px)):

        # The pixel coordinates at our point
        p1 = px[i]

        # Go through our neighbours
        for idx, n in enumerate(neighbourhood[i + 1][1:]):

            colour = [
                (255, 0, 0),     #TL
                (0, 0, 255),     #TR
                (255, 255, 255), #L
                (0, 0, 0),       #R
                (255, 255, 0),   #BL
                (255, 0, 255),   #BR
            ][idx]

            # The coordinate at our neighbours point
            if n != 0:
                p2 = px[n - 1]

                # Draw a line halfway to our target point
                p2 = p1 + ((p2 - p1) * 0.5)

                d.line([(p1[1], p1[0]), (p2[1], p2[0])], fill=colour)

    # Save our image
    os.makedirs(os.path.join('output', os.path.dirname(file)), exist_ok=True)
    img.save(os.path.join(os.path.join('output', file.replace('.jpg', '.png'))))

    import pdb
    pdb.set_trace()

    return neighbourhood


def load_data_files(files, mesh_type='VISUALMESH'):

    # Load our image and stencil
    image = tf.image.decode_jpeg(tf.read_file(files[1], name='ReadImage'), 3, name='DecodeImage')
    image = tf.image.convert_image_dtype(image, tf.float32, name='CastImage')
    stencil = tf.image.decode_png(tf.read_file(files[2], name='ReadClasses'), 1, name='DecodeClasses')

    # Read our mesh data as int32s
    mesh_data = tf.decode_raw(tf.read_file(files[0], name='ReadMesh'), tf.int32, name='DecodeMesh')

    # Work out how many mesh elements we have
    # tf.size(mesh_data) == 2(n-1) + 7(n)
    n = (tf.size(mesh_data) + 1) // 9

    # Read our pixel size as a float
    pixel_size = tf.bitcast(mesh_data[-1], tf.float32)

    # Grab our pixel coordinates and neighbourhood
    if mesh_type == 'VISUALMESH':
        pixel_coordinates = tf.reshape(mesh_data[:(n - 1) * 2], shape=[-1, 2], name='GetPixelCoordinates')

        # Get our neighbourhood indices
        neighbourhood = tf.reshape(mesh_data[(n - 1) * 2:-1], shape=[-1, 7], name='GetNeighbourhoodIndices')

    elif mesh_type in ['HEXEQUAL', 'HEXDENSE']:
        # Get our image size as floats
        img_size = tf.cast(tf.shape(stencil), tf.float32)

        if mesh_type == 'HEXEQUAL':
            c1 = tf.sqrt(tf.cast(n, tf.float32));
            c2 = tf.sqrt(img_size[1] / img_size[0]); # x/y

            nx   = tf.cast(c1 * c2, tf.int32);
            ny   = tf.cast(c1 / c2, tf.int32);
        elif mesh_type == 'HEXDENSE':
            nx = tf.cast(tf.minimum(img_size[1], img_size[1] / pixel_size), tf.int32);
            ny = tf.cast(tf.minimum(img_size[0], img_size[0] / pixel_size), tf.int32);


        # Calculate our gather matrix for fixing matrices after they have been strided
        gather_elements = ny + tf.mod(ny, 2)
        gather_rows = tf.range(gather_elements)
        gather_rows = tf.reshape(
            tf.stack([
                gather_rows[:gather_elements // 2],
                gather_rows[gather_elements // 2:],
            ],
            axis=1),
            shape=[gather_elements]
        )[:ny]

        # Make a 2d grid of our pixel coordinates
        pixel_y, pixel_x = tf.meshgrid(
            tf.linspace(0.0, img_size[0], ny + 1)[:-1],
            tf.linspace(0.0, img_size[1], nx + 1)[:-1],
            indexing='ij'
        )

        # Work out the gap between values
        dy = img_size[0] / tf.cast(ny, tf.float32)
        dx = img_size[1] / tf.cast(nx, tf.float32)

        # Offset our Y by half a jump
        pixel_y = pixel_y + dy * 0.5

        # Offset the rows by even vs odd y
        # Odd rows += 0.25 gap, even rows += 0.75 gap to make a hexagonal grid
        pixel_x = tf.gather(
            tf.concat(
                [
                    pixel_x[0::2] + (0.25 * dx),
                    pixel_x[1::2] + (0.75 * dx)
                ],
                axis=0
            ),
            gather_rows,
            axis=0
        )

        pixel_coordinates = tf.stack([pixel_y, pixel_x], axis=2)

        # Reshape into a flat list of coordinates
        pixel_coordinates = tf.reshape(pixel_coordinates, shape=[-1, 2])

        # Cast our values into ints
        pixel_coordinates = tf.cast(pixel_coordinates, tf.int32)

        # Make our neighbourhood graph
        neighbours = [None] * 7

        # First element is going to be ourself starting at index 1 to account for the offscreen point
        neighbours[0] = tf.reshape(tf.range(1, ny * nx + 1, dtype=tf.int32), shape=[ny, nx])

        # Left and right we cut off our edges and replace with 0s
        neighbours[3] = tf.pad(neighbours[0][:,:-1], [[0, 0], [1, 0]])
        neighbours[4] = tf.pad(neighbours[0][:,1:], [[0, 0], [0, 1]])

        # Interleave to get our left and right values for odd vs even rows
        left = tf.gather(
            tf.concat(
                [
                    neighbours[0][::2], # even
                    neighbours[3][1::2]  # odd
                ],
                axis=0
            ),
            gather_rows,
            axis=0
        )

        right = tf.gather(
            tf.concat(
                [
                    neighbours[4][::2], # even
                    neighbours[0][1::2]  # odd
                ],
                axis=0
            ),
            gather_rows,
            axis=0
        )

        # Row above
        neighbours[1] = tf.pad(left[:-1], [[1, 0], [0, 0]])
        neighbours[2] = tf.pad(right[:-1], [[1, 0], [0, 0]])

        # Row below
        neighbours[5] = tf.pad(left[1:], [[0, 1], [0, 0]])
        neighbours[6] = tf.pad(right[1:], [[0, 1], [0, 0]])

        # Flatten down into a list of 7 values
        neighbourhood = tf.reshape(
            tf.stack(
                neighbours,
                axis=2
            ),
            shape=[-1, 7]
        )

        # Add in our offscreen point
        neighbourhood = tf.pad(neighbourhood, [[1, 0], [0, 0]])

    # Draw the image
    neighbourhood = tf.py_func(draw_image, [files[1], pixel_coordinates, neighbourhood], tf.int32)

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

    # Prefetch 10 elements
    dataset = dataset.prefetch(10)

    return dataset.make_one_shot_iterator().string_handle()
