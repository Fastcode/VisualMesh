#!/usr/bin/env python3

import gzip
import struct
import numpy as np


def pack(file):

    # The 200 point visual mesh input
    X = []

    # The true/falseness of the check point
    Y = []

    # The index of the check point
    Yi = []

    # The adjacency graph
    G = []

    with gzip.open(file, 'rb') as f:

        while True:

            # Try to read 8 bytes
            d1 = f.read(8)
            if len(d1) < 8:
                break

            # The index of the check and the value
            ci, cv = struct.unpack('<If', d1)

            # Graph data
            graph = np.frombuffer(f.read(200 * 4 * 7), np.int32).reshape(-1, 7, order='C')

            # Colour data
            colours = np.frombuffer(f.read(200 * 4 * 3), np.float32).reshape(-1, 3, order='C')
            stencil = np.frombuffer(f.read(200 * 4 * 1), np.float32)

            # Append our various datas to the list
            X.append(colours)
            Y.append(np.array([cv, 1.0 - cv]))
            Yi.append(ci)
            G.append(graph)

    # Squash the results into numpy arrays and return
    return np.array(X), np.array(Y), np.array(Yi), np.array(G)


def validation(file):

    # THe output images
    output = []

    # Open the file
    with gzip.open(file, 'rb') as f:

        while True:

            # Try to read 8 bytes and end when there are none
            d1 = f.read(8)
            if len(d1) < 8:
                break

            # The file number and the size of this mesh
            file_no, size = struct.unpack('<II', d1)

            # Graph data (7 values per point)
            graph = np.frombuffer(f.read(size * 4 * 7), np.int32).reshape(-1, 7, order='C')

            # Pixel coordinates (2 values per point)
            coords = np.frombuffer(f.read(size * 4 * 2), np.int32).reshape(-1, 2, order='C')

            # Colour data (3 values per point)
            colours = np.frombuffer(f.read(size * 4 * 3), np.float32).reshape(-1, 3, order='C')

            # Stencil data
            stencil = np.frombuffer(f.read(size * 4 * 1), np.float32)

            output.append({
                'file_no': file_no,
                'graph': graph,
                'coords': coords,
                'colours': colours,
                'stencil': stencil
            })

    # Squash the results into numpy arrays and return
    return output
