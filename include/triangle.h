#pragma once

#include <curand_kernel.h>

#include "materials/lambertian.h"
#include "materials/material_table.h"
#include "ray.h"
#include "surface_sample.h"
#include "vec3.h"

namespace rays {

class Triangle {
public:
    __host__ __device__ Triangle(
        const Vec3 &p0,
        const Vec3 &p1,
        const Vec3 &p2,
        const Vec3 &n0,
        const Vec3 &n1,
        const Vec3 &n2,
        int materialID
    ) : m_p0(p0),
        m_p1(p1),
        m_p2(p2),
        m_n0(n0),
        m_n1(n1),
        m_n2(n2),
        m_materialID(materialID)
    {}

    __device__ bool hit(
        const Ray& ray,
        float tMin,
        float tMax,
        HitRecord& record
    ) const;

    Vec3 p0() const { return m_p0; }
    Vec3 p1() const { return m_p1; }
    Vec3 p2() const { return m_p2; }

    __host__ __device__ int materialID() const { return m_materialID; }

    __device__ SurfaceSample sample(curandState &randState) const;

    __device__ Vec3 interpolate(const float u, const float v) const {
        return (1.f - u - v) * m_p0
            + u * m_p1
            + v * m_p2;
    }

    __device__ Vec3 interpolateNormal(float u, float v) const {
        return (1.f - u - v) * m_n0
            + u * m_n1
            + v * m_n2;
    }

    __device__ float area() const {
        const Vec3 e1 = m_p1 - m_p0;
        const Vec3 e2 = m_p2 - m_p0;

        const Vec3 crossed = cross(e1, e2);
        return fabsf(crossed.length() / 2.f);
    }

    __device__ SurfaceSample sample(float xi1, float xi2) const {
        const float r1 = xi1;
        const float r2 = xi2;

        const float a = 1 - sqrt(r1);
        const float b = sqrt(r1) * (1 - r2);
        const float c = 1 - a - b;

        const Vec3 point = m_p0 * a + m_p1 * b + m_p2 * c;

        const Vec3 e1 = m_p1 - m_p0;
        const Vec3 e2 = m_p2 - m_p0;
        const Vec3 normal = normalized(cross(e1, e2));

        SurfaceSample sample = {
            .point = point,
            .normal = normal,
            .pdf = 1.f / area(),
        };
        return sample;
    }

private:
    Vec3 m_p0, m_p1, m_p2;
    Vec3 m_n0, m_n1, m_n2;

    int m_materialID;
};

}
