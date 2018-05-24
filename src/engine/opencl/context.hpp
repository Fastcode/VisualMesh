/*
 * Copyright (C) 2017-2018 Trent Houliston <trent@houliston.me>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

void setup_opencl() {

  // Get our platforms
  cl_uint platform_count = 0;
  ::clGetPlatformIDs(0, nullptr, &platform_count);
  std::vector<cl_platform_id> platforms(platform_count);
  ::clGetPlatformIDs(platforms.size(), platforms.data(), nullptr);

  // Which device/platform we are going to use
  cl_platform_id best_platform = nullptr;
  cl_device_id best_device     = nullptr;
  int best_compute_units       = 0;

  // Go through our platforms
  // for (const auto& platform : platforms) {
  const auto& platform = platforms.front();

  cl_uint device_count = 0;
  ::clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
  std::vector<cl_device_id> devices(device_count);
  ::clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count, devices.data(), nullptr);

  // Go through our devices on the platform
  for (const auto& device : devices) {

    // Length of data for strings
    size_t len;
    std::vector<char> data;

    // Print device details
    ::clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &len);
    data.resize(len);
    ::clGetDeviceInfo(device, CL_DEVICE_NAME, len, data.data(), nullptr);
    std::cout << "\tDevice: " << std::string(data.begin(), data.end()) << std::endl;


    ::clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, nullptr, &len);
    data.resize(len);
    ::clGetDeviceInfo(device, CL_DEVICE_VERSION, len, data.data(), nullptr);
    std::cout << "\tHardware version: " << std::string(data.begin(), data.end()) << std::endl;


    ::clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, nullptr, &len);
    data.resize(len);
    ::clGetDeviceInfo(device, CL_DRIVER_VERSION, len, data.data(), nullptr);
    std::cout << "\tSoftware version: " << std::string(data.begin(), data.end()) << std::endl;


    ::clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, 0, nullptr, &len);
    data.resize(len);
    ::clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, len, data.data(), nullptr);
    std::cout << "\tOpenCL C version: " << std::string(data.begin(), data.end()) << std::endl;


    cl_uint max_compute_units = 0;
    ::clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units), &max_compute_units, nullptr);
    std::cout << "\tParallel compute units: " << max_compute_units << std::endl;

    if (max_compute_units > best_compute_units) {
      best_compute_units = max_compute_units;
      best_platform      = platform;
      best_device        = device;
    }

    std::cout << std::endl;
  }

  // Print information about our selected device
  {
    // Length of data for strings
    size_t len;
    std::vector<char> data;

    // Print device details
    ::clGetDeviceInfo(best_device, CL_DEVICE_NAME, 0, nullptr, &len);
    data.resize(len);
    ::clGetDeviceInfo(best_device, CL_DEVICE_NAME, len, data.data(), nullptr);
    std::cout << "\tDevice: " << std::string(data.begin(), data.end()) << std::endl;

    ::clGetDeviceInfo(best_device, CL_DEVICE_VERSION, 0, nullptr, &len);
    data.resize(len);
    ::clGetDeviceInfo(best_device, CL_DEVICE_VERSION, len, data.data(), nullptr);
    std::cout << "\tHardware version: " << std::string(data.begin(), data.end()) << std::endl;

    ::clGetDeviceInfo(best_device, CL_DRIVER_VERSION, 0, nullptr, &len);
    data.resize(len);
    ::clGetDeviceInfo(best_device, CL_DRIVER_VERSION, len, data.data(), nullptr);
    std::cout << "\tSoftware version: " << std::string(data.begin(), data.end()) << std::endl;

    ::clGetDeviceInfo(best_device, CL_DEVICE_OPENCL_C_VERSION, 0, nullptr, &len);
    data.resize(len);
    ::clGetDeviceInfo(best_device, CL_DEVICE_OPENCL_C_VERSION, len, data.data(), nullptr);
    std::cout << "\tOpenCL C version: " << std::string(data.begin(), data.end()) << std::endl;
  }

  // Make context
  cl_int error;
  context = cl::context(::clCreateContext(nullptr, 1, &best_device, nullptr, nullptr, &error), ::clReleaseContext);
  if (error) { throw std::system_error(error, opencl_error_category(), "Error creating the OpenCL context"); }

  // Try to make an out of order queue if we can
  queue =
    cl::command_queue(::clCreateCommandQueue(context, best_device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &error),
                      ::clReleaseCommandQueue);
  if (error == CL_INVALID_VALUE) {
    queue = cl::command_queue(::clCreateCommandQueue(context, best_device, 0, &error), ::clReleaseCommandQueue);
  }
  if (error) { throw std::system_error(error, opencl_error_category(), "Error creating the OpenCL command queue"); }

  // Get program sources (this does concatenated strings)
  std::string source = PROJECT_EQUIDISTANT_CL PROJECT_EQUISOLID_CL PROJECT_RECTILINEAR_CL READ_IMAGE_TO_NETWORK_CL;
  source             = get_scalar_defines(Scalar(0.0)) + source;

  const char* cstr = source.c_str();
  size_t csize     = source.size();

  program = cl::program(::clCreateProgramWithSource(context, 1, &cstr, &csize, &error), ::clReleaseProgram);
  if (error != CL_SUCCESS) {
    throw std::system_error(error, opencl_error_category(), "Error adding sources to projection program");
  }

  // Compile the program
  error =
    ::clBuildProgram(program, 0, nullptr, "-cl-single-precision-constant -cl-fast-relaxed-math", nullptr, nullptr);
  if (error != CL_SUCCESS) {
    // Get program build log
    size_t used = 0;
    ::clGetProgramBuildInfo(program, best_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &used);
    std::vector<char> log(used);
    ::clGetProgramBuildInfo(program, best_device, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), &used);

    // Throw an error with the build log
    throw std::system_error(error,
                            opencl_error_category(),
                            "Error building projection program\n" + std::string(log.begin(), log.begin() + used));
  }

  project_rectilinear = cl::kernel(::clCreateKernel(program, "project_rectilinear", &error), ::clReleaseKernel);
  if (error != CL_SUCCESS) {
    throw std::system_error(error, opencl_error_category(), "Error getting project_rectilinear kernel");
  }
  project_equidistant = cl::kernel(::clCreateKernel(program, "project_equidistant", &error), ::clReleaseKernel);
  if (error != CL_SUCCESS) {
    throw std::system_error(error, opencl_error_category(), "Error getting project_equidistant kernel");
  }
  project_equisolid = cl::kernel(::clCreateKernel(program, "project_equisolid", &error), ::clReleaseKernel);
  if (error != CL_SUCCESS) {
    throw std::system_error(error, opencl_error_category(), "Error getting project_equisolid kernel");
  }
  read_image_to_network = cl::kernel(::clCreateKernel(program, "read_image_to_network", &error), ::clReleaseKernel);
  if (error != CL_SUCCESS) {
    throw std::system_error(error, opencl_error_category(), "Error getting read_image_to_network kernel");
  }
}
