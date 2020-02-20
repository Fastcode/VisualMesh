FROM tensorflow/tensorflow:2.1.0-gpu-py3

# Need cmake to build the op
RUN apt-get update && apt-get -y install cmake

# Need these libraries for training
RUN pip install \
    pyyaml \
    opencv-contrib-python-headless \
    matplotlib \
    tensorflow-addons

# Build the tensorflow op and put it in /visualmesh
RUN mkdir visualmesh
COPY . visualmesh/
ENV CXXFLAGS -D_GLIBCXX_USE_CXX11_ABI=0
RUN mkdir visualmesh/build && cd visualmesh/build \
    && cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLE=Off \
    -DBUILD_TENSORFLOW_OP=On \
    && make

RUN mkdir /workspace
WORKDIR /workspace
