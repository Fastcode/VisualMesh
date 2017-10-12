kernel void init_network_buffer(global const struct Node* lut, global const int* indices, global Scalar3* buffer) {

    const int index = get_global_id(0);

    // Get our real index
    const int id = indices[index];

    buffer[id] = (float3)(0, 0, 0);

    // Get our LUT node
    const struct Node n = lut[id];

    // Zero all our neighbours, we don't need to worry about ourselves since our neighbours will fill us
    for (int i = 0; i < 6; ++i) {
        buffer[id + n.neighbours[i]] = (float3)(0, 0, 0);
    }
}
