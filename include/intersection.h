#pragma once

#include "frame.h"
#include "vec3.h"

namespace rays {

struct Intersection {
    Vec3 point;
    Vec3 normal;
    Vec3 woLocal;
    Frame frame;

    __device__ bool isFront() const {
        return woLocal.z() >= 0.f;
    }
};

}
