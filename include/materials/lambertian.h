#pragma once

#include <curand_kernel.h>
#include <ostream>

#include "core/sample.h"
#include "core/vec3.h"
#include "frame.h"
#include "hit_record.h"
#include "materials/bsdf_sample.h"
#include "materials/params.h"
#include "renderers/random.h"

namespace rays {

struct Lambertian {
public:
    __host__ __device__ Lambertian() {}

    Lambertian(const MaterialParams &params)
        : m_albedo(params.getAlbedo()),
          m_emit(params.getEmit())
    {}

    __host__ __device__ Lambertian(const Vec3 &albedo)
        : m_albedo(albedo), m_emit(Vec3(0.f, 0.f, 0.f))
    {}

    __host__ __device__ Lambertian(const Vec3 &albedo, const Vec3 &emit)
        : m_albedo(albedo), m_emit(emit)
    {}

    __device__ Vec3 getEmit(const HitRecord &hit) const;
    __host__ __device__ const Vec3 &getEmit() const { return m_emit; }

    __host__ void setEmit(const Vec3 &emit);
    __host__ __device__ const Vec3 &getAlbedo() const {
        return m_albedo;
    }

    __host__ void setAlbedo(const Vec3 &albedo);

    __device__ Vec3 f(const Vec3 &wo, const Vec3 &wi) const {
        if (wo.z() < 0.f || wi.z() < 0.f) {
            return Vec3(0.f);
        }

        return m_albedo / M_PI;
    }

    __device__ float pdf(const Vec3 &wo, const Vec3 &wi) const {
        return Sample::uniformHemispherePDF(wi);
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
        const Vec3 wi = Sample::uniformHemisphere(xi1, xi2);

        return BSDFSample{
            .wiLocal = wi,
            .pdf = pdf(wo, wi),
            .f = f(wo, wi),
            .isDelta = false
        };
    }

    __device__ bool isDelta() const { return false; }

    void writeStream(std::ostream &os) const {
        os << "[Lambertian: diffuse=" << m_albedo << " emit=" << m_emit << "]";
    }

    friend std::ostream &operator<<(std::ostream &os, const Lambertian &m) {
        m.writeStream(os);
        return os;
    }

private:
    Vec3 m_albedo;
    Vec3 m_emit;
};

}
