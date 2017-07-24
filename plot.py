#!/usr/bin/env python3

import sys
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

data = ''
for l in sys.stdin:
    data += l
data = eval(data)

data = [[d[0], d[1], d[2], d[3] - d[0], d[4] - d[1], d[5] - d[2]] for d in data]
# data = [[0,0,0, d[0], d[1], d[2]] for d in data]
# data = [[0,0,0, d[3], d[4], d[5]] for d in data]

soa = np.array(data)

X, Y, Z, U, V, W = zip(*soa)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(X, Y, Z, U, V, W, arrow_length_ratio=0, linewidth=0.5)
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')


plt.show()
