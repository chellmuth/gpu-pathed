#pragma once

#include "materials/types.h"
#include "core/vec3.h"

namespace rays {

struct HitRecord {
    float t;
    Vec3 point;
    Vec3 normal;
    Vec3 wo;
    int materialID;

    __device__ bool isFront() const {
        return wo.z() >= 0.f;
    }
};

}
