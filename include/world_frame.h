#pragma once

#include "vec3.h"

namespace rays { namespace WorldFrame {
    __device__ inline float cosTheta(Vec3 normal, Vec3 w)
    {
        return fmaxf(
            0.f,
            dot(normal, w)
        );
    }

    __device__ inline float absCosTheta(Vec3 normal, Vec3 w)
    {
        return fabsf(dot(normal, w));
    }
} };
