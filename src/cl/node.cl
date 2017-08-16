// These types are shared with c++, so we need to be careful with memory alignment
#pragma pack(push, 1)

struct Node {
    /// The unit vector in the direction for this node
    Scalar4 ray;
    /// Relative indices to the linked hexagon nodes in the LUT ordered TL, TR, L, R, BL, BR,
    int neighbours[6];
};

#pragma pack(pop)
