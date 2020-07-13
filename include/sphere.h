#pragma once

#include "primitive.h"
#include "material.h"
#include "ray.h"
#include "vec3.h"

namespace rays {

class Sphere: public Primitive  {
public:
    __device__ Sphere(const Vec3 &center, float radius, Material *materialPtr)
        : m_center(center),
          m_radius(radius),
          m_materialPtr(materialPtr)
    {}

    __device__ bool hit(
        const Ray& ray,
        float tMin,
        float tMax,
        HitRecord& record
    ) const override;

private:
    Vec3 m_center;
    float m_radius;
    Material *m_materialPtr;
};

}
