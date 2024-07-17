# Copyright (C) 2017-2020 Trent Houliston <trent@houliston.me>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

find_package(PythonInterp 3 REQUIRED)

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import tensorflow; print(tensorflow.__version__)"
    OUTPUT_VARIABLE TENSORFLOW_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import tensorflow; print(tensorflow.sysconfig.get_include())"
    OUTPUT_VARIABLE tf_inc_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import tensorflow; print(tensorflow.sysconfig.get_lib())"
    OUTPUT_VARIABLE tf_lib_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE)

find_path(
    TENSORFLOW_INCLUDE_DIRS
    NAMES tensorflow/core/framework/op.h
    HINTS ${tf_inc_dir}
    DOC "TensorFlow include directory")

find_library(
    TENSORFLOW_LIBRARIES
    NAMES tensorflow_framework libtensorflow_framework.so.2 libtensorflow_framework.so.1
    HINTS ${tf_lib_dir}
    DOC "TensorFlow library")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    TensorFlow
    FOUND_VAR TensorFlow_FOUND
    REQUIRED_VARS TENSORFLOW_INCLUDE_DIRS TENSORFLOW_LIBRARIES TENSORFLOW_VERSION
    VERSION_VAR TENSORFLOW_VERSION)
