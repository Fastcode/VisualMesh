#!/usr/bin/env python3

import sys
import math
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from subprocess import run, PIPE

binary = sys.argv[1]

for s in range(3):
    for i in range(100):
        try:
            if s == 0:
                pitch = (i / 100.0) * 2.0 * math.pi
                yaw = 0.0
                roll = 0.0
            elif s == 1:
                pitch = 0.0
                yaw = (i / 100.0) * 2.0 * math.pi
                roll = 0.0
            elif s == 2:
                pitch = 0.0
                yaw = 0.0
                roll = (i / 100.0) * 2.0 * math.pi

            # Run our mesh lookup for pitch/roll/yaw
            print('Pitch: {} , Yaw: {}, Roll: {}'.format(pitch, yaw, roll))
            print('output/{:02d}-{:02d}.png'.format(s, i))
            print(' '.join([binary, '{}'.format(pitch), '{}'.format(yaw), '{}'.format(roll)]))
            result = run([binary, '{}'.format(pitch), '{}'.format(yaw), '{}'.format(roll)], stdout=PIPE)

            # Fix our data and load it as json
            data = result.stdout.decode('utf-8')
            idx = data.rfind(',')
            data = data[:idx] + data[idx+1:]
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

            plt.savefig('output/{:02d}-{:02d}.png'.format(s, i))
        except:
            print(data)
