#pragma once

#include "hit_record.h"
#include "material.h"
#include "ray.h"
#include "sphere.h"
#include "triangle.h"
#include "vec3.h"

namespace rays {

class PrimitiveList {
public:
    __device__ PrimitiveList(
        Triangle **triangles,
        size_t triangleSize,
        Sphere **spheres,
        size_t sphereSize,
        Material *materials,
        size_t materialSize
    ) : m_triangles(triangles),
        m_triangleSize(triangleSize),
        m_spheres(spheres),
        m_sphereSize(sphereSize),
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
    Triangle **m_triangles;
    size_t m_triangleSize;

    Sphere **m_spheres;
    size_t m_sphereSize;

    Material *m_materials;
    size_t m_materialSize;
};

}
