#pragma once

namespace rays {

struct Vertex {
    Vertex(float _x, float _y, float _z)
        : x(_x), y(_y), z(_z)
    {}

    float x;
    float y;
    float z;
};

struct Face {
    Face(Vertex _v0, Vertex _v1, Vertex _v2)
        : v0(_v0), v1(_v1), v2(_v2)
    {}

    Vertex v0;
    Vertex v1;
    Vertex v2;
};

}
