// These types are shared with c++, so we need to be careful with memory alignment
#pragma pack(push, 1)

struct Lens {
    enum Type { EQUIRECTANGULAR, RADIAL };
    struct Radial {
        Scalar fov;
        Scalar pixels_per_radian;
    };
    struct Equirectangular {
        Scalar2 fov;
        Scalar focal_length_pixels;
    };

    enum Type type;
    int2 dimensions;
    union {
        struct Radial radial;
        struct Equirectangular equirectangular;
    };
};

#pragma pack(pop)
