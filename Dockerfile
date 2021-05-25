# Copyright (C) 2017-2020 Trent Houliston <trent@houliston.me>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

FROM tensorflow/tensorflow:2.5.0-gpu

# Need cmake to build the op
RUN apt-get update && apt-get -y install \
    cmake \
    libtcmalloc-minimal4

# Need these libraries for training
RUN pip3 install \
    pyyaml \
    opencv-contrib-python-headless \
    matplotlib \
    tensorflow-addons \
    tqdm

# Matplotlib wants /.{config,cache}/matplotlib to be writable
RUN install -d -m 0777 /.config/matplotlib
RUN install -d -m 0777 /.cache/matplotlib

# Build the tensorflow op and put it in /visualmesh
RUN mkdir visualmesh
COPY . visualmesh/
ENV CXXFLAGS -D_GLIBCXX_USE_CXX11_ABI=0
RUN mkdir visualmesh/build && cd visualmesh/build \
    && cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLES=Off \
    -DBUILD_TENSORFLOW_OP=On \
    && make

ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4

# Make tensorflow only print out info and above logs
ENV TF_CPP_MIN_LOG_LEVEL 1

RUN mkdir /workspace
WORKDIR /workspace
