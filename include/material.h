#pragma once

#include <curand_kernel.h>

#include "frame.h"
#include "hit_record.h"
#include "vec3.h"

namespace rays {

struct Material  {
public:
    __host__ __device__ Material(const Vec3 &albedo)
        : m_albedo(albedo), m_emit(Vec3(0.f, 0.f, 0.f))
    {}

    __host__ __device__ Material(const Vec3 &albedo, const Vec3 &emit)
        : m_albedo(albedo), m_emit(emit)
    {}

    __device__ Vec3 sample(HitRecord &record, float *pdf, curandState &randState) const;
    __device__ Vec3 f(const Vec3 &wo, const Vec3 &wi) const;
    __device__ Vec3 getEmit(const HitRecord &hit) const;

    __host__ const Vec3 &getEmit() const;
    __host__ void setEmit(const Vec3 &emit);
    __host__ const Vec3 &getAlbedo() const;
    __host__ void setAlbedo(const Vec3 &albedo);

private:
    Vec3 m_albedo;
    Vec3 m_emit;
};

}
