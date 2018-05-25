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

ProjectedMesh project(const mat4& Hoc, const Lens& lens) {

  // Reused variables
  cl_int error;
  cl_event ev = nullptr;

  // Timer t;  // TIMER_LINE

  // Pack Rco into a float16
  // clang-format off
  cl_float16 Rco = {Hoc[0][0], Hoc[1][0], Hoc[2][0], Scalar(0.0),
                    Hoc[0][1], Hoc[1][1], Hoc[2][1], Scalar(0.0),
                    Hoc[0][2], Hoc[1][2], Hoc[2][2], Scalar(0.0),
                    Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0)};
  // clang-format on

  // Perform our lookup to get our relevant range
  auto ranges = lookup(Hoc, lens);

  // t.measure("\tLookup Range (cpu)");  // TIMER_LINE

  // Convenience variables
  const auto& cl_points = ranges.first.cl_points;
  const auto& nodes     = ranges.first.nodes;

  // First count the size of the buffer we will need to allocate
  int points = 0;
  for (const auto& range : ranges.second) {
    points += range.second - range.first;
  }

  // No point processing if we have no points, return an empty mesh
  if (points == 0) { return ProjectedMesh(); }

  // Build up our list of indices for OpenCL
  // Use iota to fill in the numbers
  std::vector<int> indices(points);
  auto it = indices.begin();
  for (const auto& range : ranges.second) {
    auto n = std::next(it, range.second - range.first);
    std::iota(it, n, range.first);
    it = n;
  }

  // t.measure("\tBuild Range (cpu)");  // TIMER_LINE

  // Create buffers for indices map
  cl::mem indices_map(::clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * points, nullptr, &error),
                      ::clReleaseMemObject);
  if (error) { throw std::system_error(error, opencl_error_category(), "Error allocating indices_map buffer"); }
  cl::mem pixel_coordinates(
    ::clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(std::array<Scalar, 2>) * points, nullptr, &error),
    ::clReleaseMemObject);
  if (error) { throw std::system_error(error, opencl_error_category(), "Error allocating pixel_coordinates buffer"); }

  // Upload our indices map
  cl::event indices_event;
  ev    = nullptr;
  error = ::clEnqueueWriteBuffer(
    queue, indices_map, false, 0, indices.size() * sizeof(cl_int), indices.data(), 0, nullptr, &ev);
  if (ev) indices_event = cl::event(ev, ::clReleaseEvent);
  if (error) { throw std::system_error(error, opencl_error_category(), "Error uploading indices_map to device"); }

  // indices_event.wait();               // TIMER_LINE
  // t.measure("\tUpload Range (mem)");  // TIMER_LINE

  // When everything is uploaded, we can run our projection kernel to get the pixel coordinates
  cl::event projected;
  ev = nullptr;
  /* mutex scope */ {
    std::lock_guard<std::mutex> lock(projection_mutex);

    cl::kernel projection_kernel;

    // Select a projection kernel
    switch (lens.projection) {
      case Lens::RECTILINEAR: projection_kernel = project_rectilinear; break;
      case Lens::EQUIDISTANT: projection_kernel = project_equidistant; break;
      case Lens::EQUISOLID: projection_kernel = project_equisolid; break;
    }

    // Load the arguments
    error = ::clSetKernelArg(projection_kernel, 0, cl_points.size(), &cl_points);
    if (error != CL_SUCCESS) {
      throw std::system_error(error, opencl_error_category(), "Error setting kernel argument 0 for projection kernel");
    }
    error = ::clSetKernelArg(projection_kernel, 1, indices_map.size(), &indices_map);
    if (error != CL_SUCCESS) {
      throw std::system_error(error, opencl_error_category(), "Error setting kernel argument 1 for projection kernel");
    }
    error = ::clSetKernelArg(projection_kernel, 2, sizeof(cl_float16), &Rco);
    if (error != CL_SUCCESS) {
      throw std::system_error(error, opencl_error_category(), "Error setting kernel argument 2 for projection kernel");
    }
    error = ::clSetKernelArg(projection_kernel, 3, sizeof(lens.focal_length), &lens.focal_length);
    if (error != CL_SUCCESS) {
      throw std::system_error(error, opencl_error_category(), "Error setting kernel argument 3 for projection kernel");
    }
    error = ::clSetKernelArg(projection_kernel, 4, sizeof(lens.dimensions), lens.dimensions.data());
    if (error != CL_SUCCESS) {
      throw std::system_error(error, opencl_error_category(), "Error setting kernel argument 4 for projection kernel");
    }
    error = ::clSetKernelArg(projection_kernel, 5, pixel_coordinates.size(), &pixel_coordinates);
    if (error != CL_SUCCESS) {
      throw std::system_error(error, opencl_error_category(), "Error setting kernel argument 5 for projection kernel");
    }

    // Project!
    size_t offset[1]      = {0};
    size_t global_size[1] = {size_t(points)};
    error = ::clEnqueueNDRangeKernel(queue, projection_kernel, 1, offset, global_size, nullptr, 1, &indices_event, &ev);
    if (ev) projected = cl::event(ev, ::clReleaseEvent);
    if (error != CL_SUCCESS) {
      throw std::system_error(error, opencl_error_category(), "Error queueing the projection kernel");
    }
  }
  // projected.wait();                     // TIMER_LINE
  // t.measure("\tProject points (gpu)");  // TIMER_LINE

  // This can happen on the CPU while the OpenCL device is busy
  // Build the reverse lookup map where the offscreen point is one past the end
  std::vector<int> r_indices(nodes.size(), points);
  for (uint i = 0; i < indices.size(); ++i) {
    r_indices[indices[i]] = i;
  }

  // Build the packed neighbourhood map with an extra offscreen point at the end
  std::vector<std::array<int, 6>> local_neighbourhood(points + 1);
  for (uint i = 0; i < indices.size(); ++i) {
    const auto& node = nodes[indices[i]];
    for (uint j = 0; j < 6; ++j) {
      const auto& n             = node.neighbours[j];
      local_neighbourhood[i][j] = r_indices[n];
    }
  }
  // Fill in the final offscreen point which connects only to itself
  local_neighbourhood[points].fill(points);

  // t.measure("\tBuild Local Neighbourhood (cpu)");  // TIMER_LINE

  // Create buffers for local neighbourhood
  cl::mem local_n_buffer(
    ::clCreateBuffer(
      context, CL_MEM_READ_ONLY, local_neighbourhood.size() * sizeof(std::array<int, 6>), nullptr, &error),
    ::clReleaseMemObject);
  if (error) { throw std::system_error(error, opencl_error_category(), "Error allocating local neighbourhood buffer"); }

  cl::event local_n_event;
  ev    = nullptr;
  error = ::clEnqueueWriteBuffer(queue,
                                 local_n_buffer,
                                 false,
                                 0,
                                 local_neighbourhood.size() * sizeof(std::array<int, 6>),
                                 local_neighbourhood.data(),
                                 0,
                                 nullptr,
                                 &ev);
  if (ev) local_n_event = cl::event(ev, ::clReleaseEvent);
  if (error) {
    throw std::system_error(error, opencl_error_category(), "Error uploading local neighbourhood to device");
  }

  // local_n_event.wait();                             // TIMER_LINE
  // t.measure("\tUpload Local Neighbourhood (mem)");  // TIMER_LINE
  ::clFlush(queue);

  return ProjectedMesh{LazyBufferReader<std::array<Scalar, 2>>(queue, pixel_coordinates, projected, points),
                       std::move(local_neighbourhood),
                       std::move(indices),
                       pixel_coordinates,
                       projected,
                       local_n_buffer,
                       local_n_event};
}
