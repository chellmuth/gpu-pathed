#pragma once

#include "hit_record.h"
#include "material.h"
#include "ray.h"
#include "vec3.h"

namespace rays {

class Primitive {
public:
    __device__ virtual bool hit(
        const Ray& ray,
        float tMin,
        float tMax,
        HitRecord &record
    ) const = 0;
};

class PrimitiveList {
public:
    __device__ PrimitiveList(
        Primitive **primitives,
        size_t primitiveSize,
        Material *materials,
        size_t materialSize
    ) : m_primitives(primitives),
        m_primitiveSize(primitiveSize),
        m_materials(materials),
        m_materialSize(materialSize)
    {}

    __device__ bool hit(
        const Ray& ray,
        float tMin,
        float tMax,
        HitRecord& record
    ) const;

    __device__ Material &getMaterial(size_t index) const {
        return m_materials[index];
    }

private:
    Primitive **m_primitives;
    size_t m_primitiveSize;

    Material *m_materials;
    size_t m_materialSize;
};

}
