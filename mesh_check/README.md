# Mesh Check

This tool draws a variety of meshes on images, such that the Hoc transform and the camera parameters can be checked. Checking these is not scientific, it's just an eye test to see that the mesh is drawn on the observation plane correctly.

## How to build it

Pass the flag `-DBUILD_MESH_CHECK=ON` to `cmake` when configuring, then build as usual.

## How to use it

Place your images in a directory with relative path "images", relative to the `mesh_check` binary. The images must be named "imagexxxxxxx.jpg", where xxxxxxx is a 7 digit number. They must have a lens-parameter yaml file in the same directory called "lensxxxxxxx.yaml" with the same 7 digit number.
Once your dataset is ready, run the `mesh_check` binary. It will open windows which last as long as the binary is running, drawing images with the associated meshes on each of them. If the mesh is drawn correctly on the appropriate surface, your lens-parameters are probably correct. If not, make some changes, and regenerate the lens parameters yaml file.
