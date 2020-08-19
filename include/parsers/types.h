#pragma once

#include <cmath>

namespace rays {

struct Vertex {
    Vertex(float _x, float _y, float _z)
        : x(_x), y(_y), z(_z)
    {}

    float x;
    float y;
    float z;

    Vertex operator-(const Vertex &other) const {
        return Vertex(
            x - other.x,
            y - other.y,
            z - other.z
        );
    }

    Vertex operator/(const float denominator) const {
        return Vertex(
            x / denominator,
            y / denominator,
            z / denominator
        );
    }

    bool operator==(const Vertex &other) const {
        return x == other.x && y == other.y && z == other.z;
    }


    float length() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    Vertex normalized() const {
        return *this / length();
    }

    Vertex cross(const Vertex &other) const {
        return Vertex(
            y * other.z - z * other.y,
            -(x * other.z - z * other.x),
            x * other.y - y * other.x
        );
    }
};

struct Face {
    Face(Vertex _v0, Vertex _v1, Vertex _v2,
         Vertex _n0, Vertex _n1, Vertex _n2
    ) : v0(_v0), v1(_v1), v2(_v2),
        n0(_n0), n1(_n1), n2(_n2)
    {}

    Vertex v0;
    Vertex v1;
    Vertex v2;

    Vertex n0;
    Vertex n1;
    Vertex n2;
};

struct Mtl {
    Mtl() : Mtl(0.f, 0.f, 0.f) {}

    Mtl(float _r, float _g, float _b)
        : Mtl(_r, _g, _b, 0.f, 0.f, 0.f)
    {}

    Mtl(
        float _r, float _g, float _b,
        float _emitR, float _emitG, float _emitB
    ) : r(_r), g(_g), b(_b),
        emitR(_emitR), emitG(_emitG), emitB(_emitB)
    {}

    float r;
    float g;
    float b;

    float emitR;
    float emitG;
    float emitB;
};

}
