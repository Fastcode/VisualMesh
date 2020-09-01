# Setup
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

You also need some python libraries installed using your favourite method of installing python libraries may be (e.g. pip).
```yaml
matplotlib
numpy
opencv-python
pyyaml
tensorflow
tensorflow-addons # For using the Ranger optimiser
tqdm

```

Once this is done you can run training code by using
```sh
./mesh.py train <config.yaml> <output_dir>
```

### Docker
The preferred way to build and run the training code is to use docker.
Using docker will prevent having a different version of TensorFlow etc from impacting on the running of the code.
If you are running on linux you can also forward GPUs through to docker by using nvidia-container-runtime.

```sh
# Build the docker image
docker build . --pull -t visualmesh:latest
```

This will build a docker image that can be used to run the visual mesh training and testing code.
You can then run training code by using the following command, forwarding all GPUs using the nvidia-container-runtime.

```sh
# To only use the CPU
docker run -u $(id -u):$(id -g) -it --rm --volume $(pwd):/workspace visualmesh:latest ./mesh.py  train <config.yaml> <output_dir>

# To use all available GPUs
docker run --gpus all -u $(id -u):$(id -g) -it --rm --volume $(pwd):/workspace visualmesh:latest ./mesh.py  train <config.yaml> <output_dir>

# To only use GPU0
docker run --gpus device=0 -u $(id -u):$(id -g) -it --rm --volume $(pwd):/workspace visualmesh:latest ./mesh.py  train <config.yaml> <output_dir>
