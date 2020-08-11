#pragma once

#include "core/vec3.h"
#include "util.h"

namespace rays { namespace Coordinates {

__device__ inline void cartesianToSpherical(const Vec3 &cartesian, float *phi, float *theta)
{
    *phi = atan2f(cartesian.z(), cartesian.x());
    if (*phi < 0.f) {
        *phi += 2 * M_PI;
    }
    // if (*phi == M_TWO_PI) {
    //     *phi = 0;
    // }

    *theta = acosf(util::clamp(cartesian.y(), -1.f, 1.f));
}

// __device__ Vector3 sphericalToCartesian(float phi, float theta);
// __device__ Vector3 sphericalToCartesian(float phi, float cosTheta, float sinTheta);

} }
