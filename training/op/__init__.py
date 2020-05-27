import os

import tensorflow as tf

# If we are in docker look for the visual mesh op in /visualmesh/training/visualmesh_op.so
if (
    os.path.exists("/.dockerenv")
    or os.path.isfile("/proc/self/cgroup")
    and any("docker" in line for line in open("/proc/self/cgroup"))
) and os.path.isfile("/visualmesh/training/op/visualmesh_op.so"):
    _library = tf.load_op_library("/visualmesh/training/op/visualmesh_op.so")
# Otherwise check to see if we built it and it should be right next to this file
elif os.path.isfile(os.path.join(os.path.dirname(__file__), "visualmesh_op.so")):
    _library = tf.load_op_library(os.path.join(os.path.dirname(__file__), "visualmesh_op.so"))
else:
    raise Exception("Please build the tensorflow visual mesh op before running")

ProjectVisualMesh = _library.project_visual_mesh
MapVisualMesh = _library.map_visual_mesh
UnmapVisualMesh = _library.unmap_visual_mesh
DifferenceVisualMesh = _library.difference_visual_mesh
