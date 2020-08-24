# Visual Mesh 2

[![Join the chat at https://gitter.im/Fastcode/VisualMesh](https://badges.gitter.im/Fastcode/VisualMesh.svg)](https://gitter.im/Fastcode/VisualMesh?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
If you have any questions or discussions while using the visual mesh, feel free to ask on gitter.

The Visual Mesh is an input transformation that uses knowledge a cameras orientation and position relative to an observation plane to greatly increase the performance and accuracy of a convolutional neural network.
It utilises the geometry of objects to create a mesh structure that ensures that a similar number of samples points are selected regardless of distance.
The result is that networks can be much smaller and simpler while still achieving high levels of accuracy.
Additionally it is very capable when it comes to detecting distant objects.
Normally distant objects become too small for a network to detect accurately, but as the visual mesh normalises its size detections are still accurate.

This codebase provides two components, one is a TensorFlow 2.0 based training and testing system that allows you to build and train networks.
The other component is a c++ api that is designed to operate as a high performance inference engine using OpenCL.
This allows you to get high performance on systems with devices like integrated intel GPUs with hundreds of frames per second.

## Quick Start Guide
If all you want to do is train a simple classification network for your own dataset take a look at the [Quick Start Guide](readme/quickstart.md)

## Setup
In order to setup the training and inference code you will need to provide the required libraries to build the code.

### Training/Testing
In order to train a network using the visual mesh you need to have TensorFlow 2.0+ and have built the custom op.
You have two options for how you go about this, you can either install these on your host system and train from there, or you can use the preferred method and use the provided `Dockerfile` to build a docker image you can use.

### Host System
In order to install directly on the host system you need to have CMake 3.1.0 or higher and be running in a unix like environment (linux/osx/wsl/etc).
You can't build the custom op directly on windows since TensorFlow does not provide their library on windows systems.
The dependencies for building the custom op are
- Python 3
- TensorFlow 2.X
- C++ compiler that can build c++14 code

```sh
mkdir build
cd build
cmake ..
make
```

Once this is done you can run training code by using
```sh
./mesh.py train <config.yaml> <output_dir>
```

### Docker
The prefered way to build and run the training code is to use docker.
Using docker will prevent having a different version of tensorflow etc from impacting on the running of the code.
If you are running on linux you can also forward GPUs through to docker by using nvidia-container-runtime.

```sh
# Build the docker image
docker build . --pull -t visualmesh:latest
```

This will build a docker image that can be used to run the visual mesh training and testing code.
You can then run training code by using the following command, forwarding all GPUs using the nvidia-container-runtime.

```sh
docker run --gpus all -u $(id -u):$(id -g) -it --rm --volume $(pwd):/workspace visualmesh:latest ./mesh.py  train <config.yaml> <output_dir>
```

### Inference
The inference code is based on several different engines.
Each of these engines has a different requirement in order to build code using it.
For example, the OpenCL engine requires that you have an OpenCL implementation when you build and run the code.
If you don't have these the code will still work, however you will not have access to those engines.
The specific requirements for each engine can be found in their description in the [individual sections](#Engines)

## Architecture

### Flavours
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

### Dataset
The dataset for the visual mesh should be provided as at least three TFRecord files, these being one or more for each of training, validation and testing.

Each of the flavours used in the network will add requirements for the data that needs to be provided.
You can look these up in each of the individual flavour pages above.
For example if you were to make a dataset for a Monoscopic Image Ground Classification Mesh it would need to have the following keys in it:

```python
"Hoc": float[4, 4]
"image": bytes[1] # Compressed image
"mask": bytes[1] # png image
"lens/projection": bytes[1] # 'RECTILINEAR' | 'EQUISOLID' | 'EQUIDISTANT'
"lens/focal_length": float[1]
"lens/fov": float[1]
"lens/centre": float[2]
"lens/k": float[2]
```

If you have a TFRecord dataset which has the correct data types in it, but the keys are incorrect you are able to use the keys field in order to map the keys across.
For example, if you had a field in your dataset `camera/field_of_view` which contained the data required by `lens/fov` you would be able to set up your configuration file like so.
```yaml
dataset:
  training:
    paths:
      - dataset/training.tfrecord
    keys:
      lens/fov: camera/field_of_view
```
This would allow the dataset to look for the data needed by `lens/fov` in the field `camera/field_of_view`

You are also able to override any of the settings for the flavours for only a specific dataset using a config tag.
This is typically used if you want to provide variations to your training data using the variants feature of some of the flavours, but not apply them when validating or testing the network.

For example, to manipulate the ground plane from the Ground orientation flavour only in training:

```yaml
dataset:
  training:
    paths:
      - dataset/training.tfrecord
    config:
      orientation:
        variations:
          height: { mean: 0, stddev: 0.05 }
          rotation: { mean: 0, stddev: 0.0872665 }
```

To make a simple dataset you can follow the instructions in [Quick Start Guide](readme/quickstart.md) or read the code in [training/dataset.py](training/dataset.py)

## Training
TODO discuss how to run the training

## Testing
TODO discuss how to run the testing

## Inference API
TODO discuss how to do inference of the network

### Mesh
TODO what is mesh object

### Engines
TODO what is an engine

TODO describe each engine
CPU
OpenCL
Vulkan (NOT READY YET)

### Multithreading
TODO Make multiple engines and cycle through them for best performance
