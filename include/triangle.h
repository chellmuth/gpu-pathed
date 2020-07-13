#pragma once

#include "primitive.h"
#include "material.h"
#include "vec3.h"

namespace rays {

class Triangle: public Primitive  {
public:
    __device__ Triangle(const Vec3 &p0, const Vec3 &p1, const Vec3 &p2, Material *materialPtr)
        : m_p0(p0), m_p1(p1), m_p2(p2), m_materialPtr(materialPtr)
    {}

    __device__ virtual bool hit(
        const Ray& ray,
        float tMin,
        float tMax,
        HitRecord& record
    ) const;

private:
    Material *m_materialPtr;
    Vec3 m_p0, m_p1, m_p2;
};

}
