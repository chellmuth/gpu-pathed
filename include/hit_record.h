#pragma once

#include "core/vec3.h"
#include "materials/types.h"
#include "primitives/types.h"

namespace rays {

struct HitRecord {
    float t;
    Vec3 point;
    Vec3 normal;
    Vec3 wo;
    int materialID;
    PrimitiveIndex index;

    __device__ bool isFront() const {
        return wo.z() >= 0.f;
    }
};

}
