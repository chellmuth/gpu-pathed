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

    __device__ inline float cos2Theta(const Vec3 &vector)
    {
        return vector.z() * vector.z();
    }

    __device__ inline float sin2Theta(const Vec3 &vector)
    {
        return 1.f - cos2Theta(vector);
    }

    __device__ inline float tan2Theta(const Vec3 &vector)
    {
        return sin2Theta(vector) / cos2Theta(vector);
    }

} }
