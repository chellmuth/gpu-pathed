#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "hit_record.h"
#include "materials/bsdf_sample.h"
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
        MaterialLookup *materialLookup,
        MaterialIndex *materialIndices
    ) : m_triangles(triangles),
        m_triangleSize(triangleSize),
        m_spheres(spheres),
        m_sphereSize(sphereSize),
        m_lightIndices(lightIndices),
        m_lightIndexSize(lightIndexSize),
        m_materialLookup(materialLookup),
        m_materialIndices(materialIndices)
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

    __device__ Vec3 getEmit(const int materialID) const {
        MaterialIndex index = m_materialIndices[materialID];
        return getEmit(index);
    }


    __device__ Vec3 getEmit(const int materialID, const HitRecord &record) const {
        MaterialIndex index = m_materialIndices[materialID];
        return getEmit(index, record);
    }

    __device__ Vec3 getEmit(const MaterialIndex index) const {
        switch(index.materialType) {
        case MaterialType::Lambertian: {
            return m_materialLookup->lambertians[index.index].getEmit();
        }
        case MaterialType::Mirror: {
            return m_materialLookup->mirrors[index.index].getEmit();
        }
        }
        return Vec3(0.f);
    }

    __device__ Vec3 getEmit(const MaterialIndex index, const HitRecord &record) const {
        switch(index.materialType) {
        case MaterialType::Lambertian: {
            return m_materialLookup->lambertians[index.index].getEmit(record);
        }
        case MaterialType::Mirror: {
            return m_materialLookup->mirrors[index.index].getEmit(record);
        }
        }
        return Vec3(0.f);
    }

    __device__ Vec3 f(const MaterialIndex index, const Vec3 &wo, const Vec3 &wi) const {
        switch(index.materialType) {
        case MaterialType::Lambertian: {
            return m_materialLookup->lambertians[index.index].f(wo, wi);
        }
        case MaterialType::Mirror: {
            return m_materialLookup->mirrors[index.index].f(wo, wi);
        }
        }
        return Vec3(0.f);
    }

    __device__ BSDFSample sample(
        const MaterialIndex index,
        HitRecord &record,
        curandState &randState
    ) const {
        switch(index.materialType) {
        case MaterialType::Lambertian: {
            float pdf;
            Material material = m_materialLookup->lambertians[index.index];
            Vec3 wi = material.sample(record, &pdf, randState);
            return BSDFSample{
                wi,
                pdf,
                f(index, record.wo, wi),
                material.isDelta()
            };

        }
        case MaterialType::Mirror: {
            return m_materialLookup->mirrors[index.index].sample(record, randState);
        }
        }

        return BSDFSample{};
    }

private:
    Triangle *m_triangles;
    size_t m_triangleSize;

    Sphere *m_spheres;
    size_t m_sphereSize;

    int *m_lightIndices;
    size_t m_lightIndexSize;

    MaterialLookup *m_materialLookup;
    MaterialIndex *m_materialIndices;
};

}
