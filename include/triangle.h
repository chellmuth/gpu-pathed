#pragma once

#include <curand_kernel.h>

#include "material.h"
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
        size_t materialIndex
    ) : m_p0(p0),
        m_p1(p1),
        m_p2(p2),
        m_materialIndex(materialIndex)
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

    __host__ __device__ size_t materialIndex() const { return m_materialIndex; }

    __device__ float area() const;
    __device__ SurfaceSample sample(curandState &randState) const;

    __device__ Vec3 interpolate(const float u, const float v) const {
        return (1.f - u - v) * m_p0
            + u * m_p1
            + v * m_p2;
    }

    __device__ Vec3 getNormal() const {
        const Vec3 e1 = m_p1 - m_p0;
        const Vec3 e2 = m_p2 - m_p0;

        return normalized(cross(e1, e2));
    }


private:
    Vec3 m_p0, m_p1, m_p2;

    size_t m_materialIndex;
};

}
