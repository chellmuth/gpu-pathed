#pragma once

#include <curand_kernel.h>

#include "frame.h"
#include "primitive.h"
#include "vec3.h"

namespace rays {

class Material  {
public:
    __device__ Material(const Vec3 &albedo)
        : m_albedo(albedo), m_emit(Vec3(0.f, 0.f, 0.f))
    {}

    __device__ Material(const Vec3 &albedo, const Vec3 &emit)
        : m_albedo(albedo), m_emit(emit)
    {}

    __device__ Vec3 sample(HitRecord &record, float *pdf, curandState &randState)
    {
        float z = curand_uniform(&randState);
        float r = sqrtf(fmaxf(0.f, 1.f - z * z));

        float phi = 2 * M_PI * curand_uniform(&randState);
        float x = r * cosf(phi);
        float y = r * sinf(phi);

        *pdf = 1 / (2.f * M_PI);

        return Vec3(x, y, z);
    }

    __device__ Vec3 f(const Vec3 &wo, const Vec3 &wi)
    {
        if (wo.z() < 0.f || wi.z() < 0.f) {
            return Vec3(0.f);
        }

        return m_albedo / M_PI;
    }

    __device__ Vec3 emit(const HitRecord &hit)
    {
        if (hit.isFront()) {
            return m_emit;
        }
        return Vec3(0.f);
    }

    Vec3 m_albedo;
    Vec3 m_emit;
};

}
