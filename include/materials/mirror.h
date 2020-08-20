#pragma once

#include <curand_kernel.h>

#include "frame.h"
#include "hit_record.h"
#include "materials/bsdf_sample.h"
#include "materials/params.h"
#include "core/vec3.h"

namespace rays {

struct Mirror {
public:
    __host__ __device__ Mirror() {}
    __host__ __device__ Mirror(const Vec3 &albedo) {}
    __host__ __device__ Mirror(const Vec3 &albedo, const Vec3 &emit) {}

    Mirror(const MaterialParams &params) {}

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

    __device__ float pdf(const Vec3 &wo, const Vec3 &wi) const {
        return 0.f;
    }

    __device__ BSDFSample sample(const Vec3 &wo, unsigned int &seed) const {
        return sample(wo);
    }

    __device__ BSDFSample sample(const Vec3 &wo, curandState &randState) const {
        return sample(wo);
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

    __device__ bool isDelta() const { return true; }
};

}
