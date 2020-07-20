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

    __device__ Vec3 sample(HitRecord &record, float *pdf, curandState &randState);
    __device__ Vec3 f(const Vec3 &wo, const Vec3 &wi);
    __device__ Vec3 emit(const HitRecord &hit);

private:
    Vec3 m_albedo;
    Vec3 m_emit;
};

}
