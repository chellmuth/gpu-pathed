#pragma once

#include "vec3.h"

namespace rays {

struct HitRecord {
    float t;
    Vec3 point;
    Vec3 normal;
    Vec3 wo;
    size_t materialIndex;

    __device__ bool isFront() const {
        return wo.z() >= 0.f;
    }
};

}
