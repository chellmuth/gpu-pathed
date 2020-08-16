#pragma once

#include <curand_kernel.h>

#include "core/vec3.h"
#include "materials/bsdf_sample.h"
#include "materials/params.h"
#include "renderers/random.h"


namespace rays {

class Microfacet {
public:
    __device__ Microfacet() {}

    Microfacet(const MaterialParams &params)
        : m_alpha(params.getAlpha())
    {}

    __device__ Vec3 getEmit() const { return Vec3(0.f); }

    __device__ Vec3 f(const Vec3 &wo, const Vec3 &wi) const {
        if (wo.z() < 0.f || wi.z() < 0.f) {
            return Vec3(0.f);
        }

        return Vec3(1.f) / M_PI;
    }

    __device__ BSDFSample sample(const Vec3 &wo, curandState &randState) const {
        const float xi1 = curand_uniform(&randState);
        const float xi2 = curand_uniform(&randState);
        return sample(wo, xi1, xi2);
    }

    __device__ BSDFSample sample(const Vec3 &wo, unsigned int &seed) const {
        const float xi1 = rnd(seed);
        const float xi2 = rnd(seed);
        return sample(wo, xi1, xi2);
    }

    __device__ BSDFSample sample(const Vec3 &wo, const float xi1, const float xi2) const {
        const float z = xi1;
        const float r = sqrtf(fmaxf(0.f, 1.f - z * z));

        const float phi = 2 * M_PI * xi2;
        const float x = r * cos(phi);
        const float y = r * sin(phi);

        const Vec3 wi = Vec3(x, y, z);
        return BSDFSample{
            .wiLocal = wi,
            .pdf = 1 / (2.f * M_PI),
            .f = f(wo, wi),
            .isDelta = false
        };
    }

    __device__ bool isDelta() const { return false; }

private:
    float m_alpha;
};

}
