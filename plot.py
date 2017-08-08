#!/usr/bin/env python3

import sys
import math
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from subprocess import run, PIPE

binary = sys.argv[1]

for i in range(10):
    for j in range(10):
        for k in range(10):
            try:

                pitch = i / 10.0 * 2.0 * math.pi
                yaw = j / 10.0 * 2.0 * math.pi
                roll = k / 10.0 * 2.0 * math.pi

                # Run our mesh lookup for pitch/roll/yaw
                print('Pitch: {}, Yaw: {}, Roll: {}'.format(pitch, yaw, roll))
                result = run([binary, '{}'.format(pitch), '{}'.format(yaw), '{}'.format(roll)], stdout=PIPE)

                # Fix our data and load it as json
                data = '{}]'.format(result.stdout.decode('utf-8')[:-5])
                data = json.loads(data)
                print('\tFound {} points'.format(len(data)))

                # Fix our data
                data = np.array([[d[0], d[1], d[2], d[3] - d[0], d[4] - d[1], d[5] - d[2]] for d in data])

                X, Y, Z, U, V, W = zip(*data)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.quiver(X, Y, Z, U, V, W, arrow_length_ratio=0, linewidth=0.5)
                ax.set_xlim([-1, 1])
                ax.set_ylim([-1, 1])
                ax.set_zlim([-1, 1])
                ax.set_xlabel('X axis')
                ax.set_ylabel('Y axis')
                ax.set_zlabel('Z axis')

                plt.savefig('output/{:01d}-{:01d}-{:01d}.png'.format(i, j, k))
            except:
                pass
