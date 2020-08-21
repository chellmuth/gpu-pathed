#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "core/ray.h"
#include "core/vec3.h"
#include "frame.h"
#include "hit_record.h"
#include "lights/environment_light.h"
#include "lights/sampler.h"
#include "lights/types.h"
#include "materials/bsdf.h"
#include "materials/bsdf_sample.h"
#include "materials/lambertian.h"
#include "materials/material_lookup.h"
#include "primitives/sphere.h"
#include "primitives/triangle.h"
#include "primitives/types.h"

namespace rays {

class World {
public:
    __device__ World(
        Triangle *triangles,
        size_t triangleSize,
        Sphere *spheres,
        size_t sphereSize,
        LightIndex *lightIndices,
        size_t lightIndexSize,
        EnvironmentLight *environmentLight,
        MaterialLookup *materialLookup
    ) : m_triangles(triangles),
        m_triangleSize(triangleSize),
        m_spheres(spheres),
        m_sphereSize(sphereSize),
        m_lightIndices(lightIndices),
        m_lightIndexSize(lightIndexSize),
        m_environmentLight(environmentLight),
        m_materialLookup(materialLookup)
    {}

    __device__ bool hit(
        const Ray& ray,
        float tMin,
        float tMax,
        HitRecord& record
    ) const;

    __device__ LightSample sampleDirectLights(
        const Vec3 &hitPoint,
        const Frame &frame,
        curandState &randState
    ) const {
        const float xi1 = curand_uniform(&randState);
        const float xi2 = curand_uniform(&randState);
        const float xi3 = curand_uniform(&randState);

        return Sampler::sampleDirectLights(
            hitPoint,
            frame,
            make_float3(xi1, xi2, xi3),
            m_lightIndices,
            m_lightIndexSize,
            m_triangles,
            m_spheres,
            *m_environmentLight,
            *m_materialLookup
        );
    }

    __device__ float pdfSceneLights(
        const Vec3 &referencePoint,
        const HitRecord &lightRecord
    ) const {
        const float lightPDF = rays::Sampler::pdfSceneLights(
            referencePoint,
            lightRecord.point,
            lightRecord.normal,
            lightRecord.index,
            m_lightIndices,
            m_lightIndexSize,
            m_triangles,
            m_spheres,
            *m_environmentLight
        );
        return lightPDF;
    }

    __device__ float pdfEnvironmentLight(const Vec3 &wi) const {
        const float lightPDF = rays::Sampler::pdfEnvironmentLight(
            wi,
            *m_environmentLight,
            m_lightIndexSize
        );
        return lightPDF;
    }

    __device__ float pdfBSDF(
        int materialID,
        const Vec3 &wo,
        const Vec3 &wi
    ) const {
        BSDF bsdf(m_materialLookup, materialID);
        return bsdf.pdf(wo, wi);
    }

    __device__ Vec3 getEmit(const int materialID) const {
        return m_materialLookup->getEmit(materialID);
    }

    __device__ Vec3 getEmit(const int materialID, const HitRecord &record) const {
        MaterialIndex index = m_materialLookup->indices[materialID];
        return getEmit(index, record);
    }

    __device__ Vec3 getEmit(const MaterialIndex index, const HitRecord &record) const {
        switch(index.materialType) {
        case MaterialType::Lambertian: {
            return m_materialLookup->lambertians[index.index].getEmit(record);
        }
        case MaterialType::Mirror: {
            return m_materialLookup->mirrors[index.index].getEmit(record);
        }
        case MaterialType::Glass: {
            return m_materialLookup->glasses[index.index].getEmit(record);
        }
        }
        return Vec3(0.f);
    }

    __device__ Vec3 f(const int materialID, const Vec3 &wo, const Vec3 &wi) const {
        BSDF bsdf(m_materialLookup, materialID);
        return bsdf.f(wo, wi);
    }

    __device__ BSDFSample sample(
        int materialID,
        HitRecord &record,
        curandState &randState
    ) const {
        BSDF bsdf(m_materialLookup, materialID);
        return bsdf.sample(record, randState);
    }

    __device__ Vec3 environmentL(const Vec3 &wi) const {
        return m_environmentLight->getEmit(wi);
    }

private:
    Triangle *m_triangles;
    size_t m_triangleSize;

    Sphere *m_spheres;
    size_t m_sphereSize;

    LightIndex *m_lightIndices;
    size_t m_lightIndexSize;

    EnvironmentLight *m_environmentLight;

    MaterialLookup *m_materialLookup;
};

}
