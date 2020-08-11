#pragma once

#include "core/vec3.h"

namespace rays { namespace Coordinates {

__device__ void cartesianToSpherical(const Vec3 &cartesian, float *phi, float *theta);
// __device__ Vector3 sphericalToCartesian(float phi, float theta);
// __device__ Vector3 sphericalToCartesian(float phi, float cosTheta, float sinTheta);

} }
