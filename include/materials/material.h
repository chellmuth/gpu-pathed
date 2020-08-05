#pragma once

#include <curand_kernel.h>
#include <ostream>

#include "frame.h"
#include "hit_record.h"
#include "vec3.h"

namespace rays {

struct Material  {
public:
    __host__ __device__ Material() {}

    __host__ __device__ Material(const Vec3 &albedo)
        : m_albedo(albedo), m_emit(Vec3(0.f, 0.f, 0.f))
    {}

    __host__ __device__ Material(const Vec3 &albedo, const Vec3 &emit)
        : m_albedo(albedo), m_emit(emit)
    {}

    __device__ Vec3 sample(HitRecord &record, float *pdf, curandState &randState) const;

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

    __device__ Vec3 sample(float *pdf, const float2 xis) const {
        const float z = xis.x;
        const float r = sqrtf(fmaxf(0.f, 1.f - z * z));

        const float phi = 2 * M_PI * xis.y;
        const float x = r * cos(phi);
        const float y = r * sin(phi);

        *pdf = 1 / (2.f * M_PI);

        return Vec3(x, y, z);
    }

    __device__ bool isDelta() const { return false; }

    void writeStream(std::ostream &os) const {
        os << "[Lambertian: diffuse=" << m_albedo << " emit=" << m_emit << "]";
    }

    friend std::ostream &operator<<(std::ostream &os, const Material &m) {
        m.writeStream(os);
        return os;
    }

private:
    Vec3 m_albedo;
    Vec3 m_emit;
};

}
