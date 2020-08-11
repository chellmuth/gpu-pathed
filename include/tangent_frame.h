#pragma once

#include "core/vec3.h"

namespace rays { namespace TangentFrame {
    __device__ inline float cosTheta(const Vec3 &vector)
    {
        return vector.z();
    }

    __device__ inline float absCosTheta(const Vec3 &vector)
    {
        return fabsf(vector.z());
    }

} }
