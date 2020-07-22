#pragma once

#include "material.h"
#include "ray.h"
#include "vec3.h"

namespace rays {

class Sphere  {
public:
    __host__ __device__ Sphere(
        const Vec3 &center,
        float radius,
        size_t materialIndex
    )
        : m_center(center),
          m_radius(radius),
          m_materialIndex(materialIndex)
    {}

    __device__ bool hit(
        const Ray& ray,
        float tMin,
        float tMax,
        HitRecord& record
    ) const;

private:
    Vec3 m_center;
    float m_radius;
    size_t m_materialIndex;
};

}
