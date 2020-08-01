#include "materials/dummy.h"

namespace rays {

__device__ Vec3 Dummy::sample(
    HitRecord &record,
    float *pdf,
    curandState &randState
) const {
    float z = curand_uniform(&randState);
    float r = sqrt(max(0.f, 1.f - z * z));

    float phi = 2 * M_PI * curand_uniform(&randState);
    float x = r * cos(phi);
    float y = r * sin(phi);

    *pdf = 1 / (2.f * M_PI);

    return Vec3(x, y, z);
}

__device__ Vec3 Dummy::getEmit(const HitRecord &hit) const
{
    return Vec3(0.f);
}

__host__ void Dummy::setEmit(const Vec3 &emit)
{
}

__host__ void Dummy::setAlbedo(const Vec3 &albedo)
{
}


}
