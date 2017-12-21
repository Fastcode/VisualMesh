#!/usr/bin/env python3

import gzip
import struct
import io
import subprocess
import numpy as np

def pack(file):

    # The 200 point visual mesh input
    X = []

    # The true/falseness of the check point
    Y = []

    # The adjacency graph
    G = []

    p = subprocess.Popen(['gzcat', file], stdout=subprocess.PIPE)
    with io.BytesIO(p.communicate()[0]) as f:

        while True:
            size_data = f.read(4)

            # When we run out of data, stop
            if len(size_data) != 4:
                break

            size, = struct.unpack('<I', size_data)

            # Read the graph and adjust for new indexing
            graph = np.frombuffer(f.read(size * 4 * 7), np.int32).reshape(-1, 7, order='C')
            graph = graph + 1

            # Insert our 0 row
            graph = np.insert(graph, 0, [0] * 7, axis=0)

            # Get colours and insert 0 row
            colours = np.frombuffer(f.read(size * 4 * 3), np.float32).reshape(-1, 3, order='C')
            colours = np.insert(colours, 0, [0] * 3, axis=0)

            # Get stencil and insert 0 row
            stencil = np.frombuffer(f.read(size * 4), np.float32)
            stencil = np.insert(stencil, 0, [0] * 1, axis=0)

            X.append(colours)
            Y.append(stencil)
            G.append(graph)

    # Return the loaded data
    return X, Y, G


def tree(file):

    # The 200 point visual mesh input
    X = []

    # The true/falseness of the check point
    Y = []

    # The index of the check point
    Yi = []

    # The adjacency graph
    G = []

    p = subprocess.Popen(['lz4', '-dc', file], stdout=subprocess.PIPE)
    with io.BytesIO(p.communicate()[0]) as f:

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
    p = subprocess.Popen(['lz4', '-dc', file], stdout=subprocess.PIPE)
    with io.BytesIO(p.communicate()[0]) as f:

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
