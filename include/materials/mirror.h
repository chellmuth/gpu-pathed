#pragma once

#include <curand_kernel.h>

#include "frame.h"
#include "hit_record.h"
#include "materials/bsdf_sample.h"
#include "vec3.h"

namespace rays {

struct Mirror  {
public:
    __host__ __device__ Mirror() {}

    __host__ __device__ Mirror(const Vec3 &albedo)
    {}

    __host__ __device__ Mirror(const Vec3 &albedo, const Vec3 &emit)
    {}

    __device__ Vec3 getEmit(const HitRecord &hit) const;
    __host__ __device__ Vec3 getEmit() const { return Vec3(0.f); }

    __host__ void setEmit(const Vec3 &emit);
    __host__ __device__ Vec3 getAlbedo() const {
        return Vec3(0.f);
    }

    __host__ void setAlbedo(const Vec3 &albedo);

    __device__ Vec3 f(const Vec3 &wo, const Vec3 &wi) const {
        return Vec3(0.f);
    }

    __device__ BSDFSample sample(HitRecord &record, curandState &randState) const {
        Vec3 wi = record.wo.reflect(Vec3(0.f, 0.f, 1.f));
        return BSDFSample{
            wi,
            1.f,
            Vec3(fmaxf(0.f, 1.f / wi.z())),
            isDelta()
        };
    }

    __device__ BSDFSample sample(const Vec3 &wo) const {
        const Vec3 wi = wo.reflect(Vec3(0.f, 0.f, 1.f));
        return BSDFSample{
            wi,
            1.f,
            Vec3(fmaxf(0.f, 1.f / wi.z())),
            isDelta()
        };
    }

    __device__ Vec3 sample(const Vec3 &wo, float *pdf) const {
        const Vec3 wi = wo.reflect(Vec3(0.f, 0.f, 1.f));
        *pdf = 1.f;
        return wi;
    }

    __device__ bool isDelta() const { return true; }
};

}
