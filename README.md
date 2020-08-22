# Visual Mesh 2

[![Join the chat at https://gitter.im/Fastcode/VisualMesh](https://badges.gitter.im/Fastcode/VisualMesh.svg)](https://gitter.im/Fastcode/VisualMesh?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
If you have any questions or discussions while using the visual mesh, feel free to ask on gitter.

The Visual Mesh is an input transformation that uses knowledge a cameras orientation and position relative to an observation plane to greatly increase the performance and accuracy of a convolutional neural network.
It utilises the geometry of objects to create a mesh structure that ensures that a similar number of samples points are selected regardless of distance.
The result is that networks can be much smaller and simpler while still achieving high levels of accuracy.
Additionally it is very capable when it comes to detecting distant objects.
Normally distant objects become too small for a network to detect accurately, but as the visual mesh normalises its size detections are still accurate.

## Quickstart Guide
If all you want to do is train a simple classification network for your own dataset take a look at the [Quickstart Guide](readme/quickstart.md)

## Setup
In order to train a network using the visual mesh you need to have tensorflow 2.X and have built the custom op.
You have two options for how you go about this, you can either install these on your host system and train from there, or you can use the preferred method and use the provided `Dockerfile` to build a docker image you can use.

### Custom Op
TODO Cmake build etc

### Docker
TODO Docker
TODO Tensorflow 2.0

## Flavours
The Visual Mesh code is designed to easily be able to implement new ideas for how to use it.
It allows this by providing several different components that can be combined to make a vision system.
In the code these components are refered to as flavours.
There are five different categories for these that are each discussed on their own pages.

[View](readme/flavour/view.md) describes how the system is set up in terms of cameras.
It can be used to implement features like Stereoscopic or Multi Camera systems.

[Orientation](readme/flavour/orientation.md) describes how the visual mesh's observation plane is set up from the perspective of the camera.
For example the traditional position for the observation plane is on the ground below the camera.
However if you were to attach the plane to another flat surface this flavour can provide the tools to do that.

[Projection](readme/flavour/projection.md) describes how the visual mesh itself is designed and projected into pixel coordinates.
It includes details about which visual mesh model to use and also contains the logic that uses lens parameters to project to image space.

[Example](readme/flavour/example.md) describes how the training input examples are provided.
For example, the Image method will load images from the dataset and grab the pixels projected by the visual mesh.
If you find some other wonderful way to use the visual mesh besides images put it here!

[Label](readme/flavour/label.md) describes how the training labels are provided.
For a classification network this would be the mask image that the labels are gathered from.

## Dataset

The dataset for the visual mesh should be provided as three `.tfrecord` files.
Each of these flavours will have requirements for the data that needs to be provided.
For example if you a classification dataset

## Training
TODO discuss how to run the training

## Testing
TODO discuss how to run the testing

## Execution
TODO discuss how to do inference of the network

### C++ api

#### Mesh
TODO what is mesh object

#### Engines
TODO what is an engine

TODO describe each engine
CPU
OpenCL
Vulkan (NOT READY YET)

#### Multithreading
TODO Make multiple engines and cycle through them for best performance
