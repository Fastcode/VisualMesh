kernel void project_equirectangular(global const struct Node* lut,
                                    global const int* indices,
                                    global const Scalar3* Rco,
                                    const struct Lens lens,
                                    global int2* out) {

    const int index = get_global_id(0);

    // Get our real index
    const int id = indices[index];

    // Get our LUT node
    const struct Node n = lut[id];

    // Rotate our ray by our matrix to put it in the camera space
    const Scalar3 ray = (Scalar3)(dot(Rco[0], n.ray), dot(Rco[1], n.ray), dot(Rco[2], n.ray));

    // Work out our pixel coordinates as a 0 centred image with x to the left and y up (screen space)
    const Scalar2 screen = (Scalar2)(lens.equirectangular.focal_length_pixels * ray.y / ray.x,
                                     lens.equirectangular.focal_length_pixels * ray.z / ray.x);

    // Apply our offset to move into image space (0 at top left, x to the right, y down)
    const Scalar2 image =
        (Scalar2)((Scalar)(lens.dimensions.x - 1) * 0.5, (Scalar)(lens.dimensions.y - 1) * 0.5) - screen;

    // Store our output coordinates
    out[index] = (int2)(round(image[0]), round(image[1]));
}
