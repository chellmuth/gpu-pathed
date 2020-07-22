#pragma once

#include "material.h"
#include "ray.h"
#include "vec3.h"

namespace rays {

class Triangle  {
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

private:
    Vec3 m_p0, m_p1, m_p2;

    size_t m_materialIndex;
};

}
