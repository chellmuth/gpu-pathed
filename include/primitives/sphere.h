#pragma once

#include "materials/lambertian.h"
#include "core/ray.h"
#include "core/vec3.h"

namespace rays {

class Sphere  {
public:
    __host__ __device__ Sphere(
        const Vec3 &center,
        float radius,
        int materialID
    )
        : m_center(center),
          m_radius(radius),
          m_materialID(materialID)
    {}

    __device__ bool hit(
        const Ray& ray,
        float tMin,
        float tMax,
        HitRecord& record
    ) const;

    __host__ __device__ int materialID() const { return m_materialID; }

    __host__ __device__ Vec3 getCenter() const {
        return m_center;
    }

    __host__ __device__ float getRadius() const {
        return m_radius;
    }

private:
    Vec3 m_center;
    float m_radius;
    int m_materialID;
};

}
