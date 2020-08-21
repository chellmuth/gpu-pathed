#pragma once

#include "core/vec3.h"
#include "frame.h"
#include "primitives/types.h"

namespace rays {

struct Intersection {
    Vec3 point;
    Vec3 normal;
    Vec3 woLocal;
    Frame frame;
    PrimitiveIndex index;

    __device__ bool isFront() const {
        return woLocal.z() >= 0.f;
    }
};

}
