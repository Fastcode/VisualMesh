# Projection functions should take an orientation and other properties and return projected coordinates
# The returned data should contain at least the following
# {
#   'G': The graph used to connect the results with -1 used as the off screen point
#   'C': The pixel coordinates of each the visual mesh nodes
#   'V': The unit vectors from the camera in observation plane space the pixels represent
#   'I': The global index of this point used for stereo matching
# }

from .visual_mesh import VisualMesh
