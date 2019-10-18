find_package(PythonInterp 3 REQUIRED)

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "import tensorflow; print(tensorflow.__version__)"
  OUTPUT_VARIABLE TENSORFLOW_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "import tensorflow; print(tensorflow.sysconfig.get_include())"
  OUTPUT_VARIABLE tf_inc_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "import tensorflow; print(tensorflow.sysconfig.get_lib())"
  OUTPUT_VARIABLE tf_lib_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_path(
  TENSORFLOW_INCLUDE_DIRS
  NAMES tensorflow/core/framework/op.h
  HINTS ${tf_inc_dir}
  DOC "TensorFlow include directory"
)

find_library(
  TENSORFLOW_LIBRARIES
  NAMES tensorflow_framework libtensorflow_framework.so.2 libtensorflow_framework.so.1
  HINTS ${tf_lib_dir}
  DOC "TensorFlow library"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  TensorFlow
  FOUND_VAR
  TensorFlow_FOUND
  REQUIRED_VARS
  TENSORFLOW_INCLUDE_DIRS
  TENSORFLOW_LIBRARIES
  TENSORFLOW_VERSION
  VERSION_VAR
  TENSORFLOW_VERSION
)
