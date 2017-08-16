kernel void project_radial(global struct Node* lut,
                           global int* indices,
                           global Scalar4* Rco,
                           const struct Lens lens,
                           global int2* out) {

    const int index = get_global_id(0);

    // Get our real index
    const int id = indices[index];

    // Get our LUT node
    const struct Node n = lut[id];

    // Rotate our ray by our matrix to put it into camera space
    const Scalar3 ray = (Scalar3)(dot(Rco[0], n.ray), dot(Rco[1], n.ray), dot(Rco[2], n.ray));

    // Calculate some intermediates
    const Scalar theta     = acos(n.ray[0]);
    const Scalar r         = theta * lens.radial.pixels_per_radian;
    const Scalar sin_theta = sin(theta);

    // Work out our pixel coordinates as a 0 centred image with x to the left and y up (screen space)
    const Scalar2 screen = (Scalar2)(r * ray.y / sin_theta, r * ray.z / sin_theta);

    // Apply our offset to move into image space (0 at top left, x to the right, y down)
    const Scalar2 image =
        (Scalar2)((Scalar)(lens.dimensions.x - 1) * 0.5, (Scalar)(lens.dimensions.y - 1) * 0.5) - screen;

    // Apply our lens centre offset
    // TODO apply this

    // Store our output coordinates
    out[index] = (int2)(round(image[0]), round(image[1]));
}
