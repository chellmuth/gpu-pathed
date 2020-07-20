#include "material.h"

#include <curand_kernel.h>

namespace rays {

__device__ Vec3 Material::sample(HitRecord &record, float *pdf, curandState &randState)
{
    float z = curand_uniform(&randState);
    float r = sqrt(max(0.f, 1.f - z * z));

    float phi = 2 * M_PI * curand_uniform(&randState);
    float x = r * cos(phi);
    float y = r * sin(phi);

    *pdf = 1 / (2.f * M_PI);

    return Vec3(x, y, z);
}

__device__ Vec3 Material::f(const Vec3 &wo, const Vec3 &wi)
{
    if (wo.z() < 0.f || wi.z() < 0.f) {
        return Vec3(0.f);
    }

    return m_albedo / M_PI;
}

__device__ Vec3 Material::emit(const HitRecord &hit)
{
    if (hit.isFront()) {
        return m_emit;
    }
    return Vec3(0.f);
}

}
