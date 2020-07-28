#pragma once

#include <curand_kernel.h>

#include "hit_record.h"
#include "material.h"
#include "ray.h"
#include "sphere.h"
#include "triangle.h"
#include "vec3.h"
#include "world_frame.h"

namespace rays {

struct LightSample {
    Vec3 point;
    Vec3 normal;
    float pdf;
    size_t materialIndex;

    __device__ float solidAnglePDF(const Vec3 &referencePoint) const
    {
        const Vec3 lightDirection = point - referencePoint;
        const Vec3 lightWo = -normalized(lightDirection);
        const float distance = lightDirection.length();

        const float distance2 = distance * distance;
        const float projectedArea = WorldFrame::cosTheta(normal, lightWo);

        return pdf * distance2 / projectedArea;
    }

};

class PrimitiveList {
public:
    __device__ PrimitiveList(
        Triangle *triangles,
        size_t triangleSize,
        Sphere *spheres,
        size_t sphereSize,
        int *lightIndices,
        size_t lightIndexSize,
        Material *materials,
        size_t materialSize
    ) : m_triangles(triangles),
        m_triangleSize(triangleSize),
        m_spheres(spheres),
        m_sphereSize(sphereSize),
        m_lightIndices(lightIndices),
        m_lightIndexSize(lightIndexSize),
        m_materials(materials),
        m_materialSize(materialSize)
    {}

    __device__ bool hit(
        const Ray& ray,
        float tMin,
        float tMax,
        HitRecord& record
    ) const;

    __device__ LightSample sampleDirectLights(Vec3 hitPoint, curandState &randState) const {
        const int lightChoice = (int)floorf(curand_uniform(&randState) * m_lightIndexSize);
        const float choicePDF = 1.f / m_lightIndexSize;

        const Triangle &triangle = m_triangles[m_lightIndices[lightChoice]];
        const SurfaceSample sample = triangle.sample(randState);

        LightSample lightSample = {
            sample.point,
            sample.normal,
            sample.pdf * choicePDF,
            triangle.materialIndex()
        };
        return lightSample;
    }

    __device__ Material &getMaterial(size_t index) const {
        return m_materials[index];
    }

private:
    Triangle *m_triangles;
    size_t m_triangleSize;

    Sphere *m_spheres;
    size_t m_sphereSize;

    int *m_lightIndices;
    size_t m_lightIndexSize;

    Material *m_materials;
    size_t m_materialSize;
};

}
