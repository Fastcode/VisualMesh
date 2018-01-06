const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

enum FOURCC {
    GREY    = 0x59455247,
    Y12     = 0x20323159,
    Y16     = 0x20363159,
    GRBG    = 0x47425247,
    RGGB    = 0x42474752,
    GBRG    = 0x47524247,
    BGGR    = 0x52474742,
    GR12    = 0x32315247,
    RG12    = 0x32314752,
    GB12    = 0x32314247,
    BG12    = 0x32314742,
    GR16    = 0x36315247,
    RG16    = 0x36314752,
    GB16    = 0x36314247,
    BG16    = 0x36314742,
    Y411    = 0x31313459,
    UYVY    = 0x59565955,
    YUYV    = 0x56595559,
    YM24    = 0x34324d59,
    RGB3    = 0x33424752,
    RGBA    = 0x41424752,
    BGR3    = 0x33524742,
    BGRA    = 0x41524742,
    JPEG    = 0x4745504a,
    UNKNOWN = 0
};

Scalar4 read_image(read_only image2d_t image, const enum FOURCC format, const int2 coordinates) {
    switch (format) {
        case YUYV: {
            return (Scalar4)(1.0);  // TODO implement
        }
        case RGB3:
        case BGRA:
        case RGBA: {
            return read_imagef(image, sampler, coordinates);
        }
        default: { return (Scalar4)(0); }
    }
}

kernel void read_image_to_network(read_only image2d_t image,
                                  const enum FOURCC format,
                                  global int2* coordinates,
                                  global Scalar4* network) {

    const int idx = get_global_id(0);

    // Store our pixel value in the network
    network[idx] = read_image(image, format, coordinates[idx]);
}
