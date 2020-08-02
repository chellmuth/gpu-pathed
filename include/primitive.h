#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "hit_record.h"
#include "materials/material.h"
#include "materials/material_table.h"
#include "ray.h"
#include "sampler.h"
#include "sphere.h"
#include "triangle.h"
#include "vec3.h"

namespace rays {

class PrimitiveList {
public:
    __device__ PrimitiveList(
        Triangle *triangles,
        size_t triangleSize,
        Sphere *spheres,
        size_t sphereSize,
        int *lightIndices,
        size_t lightIndexSize,
        MaterialLookup *materialLookup
    ) : m_triangles(triangles),
        m_triangleSize(triangleSize),
        m_spheres(spheres),
        m_sphereSize(sphereSize),
        m_lightIndices(lightIndices),
        m_lightIndexSize(lightIndexSize),
        m_materialLookup(materialLookup)
    {}

    __device__ bool hit(
        const Ray& ray,
        float tMin,
        float tMax,
        HitRecord& record
    ) const;

    __device__ LightSample sampleDirectLights(Vec3 hitPoint, curandState &randState) const {
        const float xi1 = curand_uniform(&randState);
        const float xi2 = curand_uniform(&randState);
        const float xi3 = curand_uniform(&randState);

        return Sampler::sampleDirectLights(
            hitPoint,
            make_float3(xi1, xi2, xi3),
            m_lightIndices,
            m_lightIndexSize,
            m_triangles
        );
    }

    __device__ Material &getMaterial(MaterialIndex index) const {
        return m_materialLookup->lambertians[index.index]; // fixme
    }

private:
    Triangle *m_triangles;
    size_t m_triangleSize;

    Sphere *m_spheres;
    size_t m_sphereSize;

    int *m_lightIndices;
    size_t m_lightIndexSize;

    MaterialLookup *m_materialLookup;
};

}
