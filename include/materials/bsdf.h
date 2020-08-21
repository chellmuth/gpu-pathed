#pragma once

#include <curand_kernel.h>

#include "intersection.h"
#include "materials/bsdf_sample.h"
#include "materials/material_lookup.h"
#include "renderers/random.h"

namespace rays {

class BSDF {
public:
    __device__ BSDF(const MaterialLookup *materialLookupPtr, int materialID)
        : m_materialLookupPtr(materialLookupPtr),
          m_materialID(materialID)
    {}

    __device__ Vec3 f(const Vec3 &wo, const Vec3 &wi) {
        const MaterialIndex index = m_materialLookupPtr->indices[m_materialID];
        switch(index.materialType) {
        case MaterialType::Lambertian: {
            return m_materialLookupPtr->lambertians[index.index].f(wo, wi);
        }
        case MaterialType::Mirror: {
            return m_materialLookupPtr->mirrors[index.index].f(wo, wi);
        }
        case MaterialType::Glass: {
            return m_materialLookupPtr->glasses[index.index].f(wo, wi);
        }
        case MaterialType::Microfacet: {
            return m_materialLookupPtr->microfacets[index.index].f(wo, wi);
        }
        }
        return Vec3(0.f);
    }

    __device__ float pdf(const Vec3 &wo, const Vec3 &wi) {
        const MaterialIndex index = m_materialLookupPtr->indices[m_materialID];
        switch(index.materialType) {
        case MaterialType::Lambertian: {
            return m_materialLookupPtr->lambertians[index.index].pdf(wo, wi);
        }
        case MaterialType::Mirror: {
            return m_materialLookupPtr->mirrors[index.index].pdf(wo, wi);
        }
        case MaterialType::Glass: {
            return m_materialLookupPtr->glasses[index.index].pdf(wo, wi);
        }
        case MaterialType::Microfacet: {
            return m_materialLookupPtr->microfacets[index.index].pdf(wo, wi);
        }
        }
        return 0.f;
    }

    __device__ BSDFSample sample(
        const Intersection &intersection,
        unsigned int &seed
    ) {
        BSDFSample sample;

        const MaterialIndex index = m_materialLookupPtr->indices[m_materialID];
        switch(index.materialType) {
        case MaterialType::Lambertian: {
            sample = m_materialLookupPtr->lambertians[index.index].sample(intersection.woLocal, seed);
            break;
        }
        case MaterialType::Mirror: {
            sample = m_materialLookupPtr->mirrors[index.index].sample(intersection.woLocal, seed);
            break;
        }
        case MaterialType::Glass: {
            sample = m_materialLookupPtr->glasses[index.index].sample(intersection.woLocal, seed);
            break;
        }
        case MaterialType::Microfacet: {
            sample = m_materialLookupPtr->microfacets[index.index].sample(intersection.woLocal, seed);
            break;
        }
        }

        sample.materialID = m_materialID;
        return sample;
    }

    __device__ BSDFSample sample(
        const HitRecord &record,
        curandState &randState
    ) {
        BSDFSample sample;

        const MaterialIndex index = m_materialLookupPtr->indices[m_materialID];
        switch(index.materialType) {
        case MaterialType::Lambertian: {
            sample = m_materialLookupPtr->lambertians[index.index].sample(record.wo, randState);
            break;
        }
        case MaterialType::Mirror: {
            sample = m_materialLookupPtr->mirrors[index.index].sample(record.wo, randState);
            break;
        }
        case MaterialType::Glass: {
            sample = m_materialLookupPtr->glasses[index.index].sample(record.wo, randState);
            break;
        }
        case MaterialType::Microfacet: {
            sample = m_materialLookupPtr->microfacets[index.index].sample(record.wo, randState);
            break;
        }
        }

        sample.materialID = m_materialID;
        return sample;
    }

    __device__ Vec3 getEmit() {
        const MaterialIndex index = m_materialLookupPtr->indices[m_materialID];
        switch(index.materialType) {
        case MaterialType::Lambertian: {
            return m_materialLookupPtr->lambertians[index.index].getEmit();
        }
        case MaterialType::Mirror: {
            return m_materialLookupPtr->mirrors[index.index].getEmit();
        }
        case MaterialType::Glass: {
            return m_materialLookupPtr->glasses[index.index].getEmit();
        }
        case MaterialType::Microfacet: {
            return m_materialLookupPtr->microfacets[index.index].getEmit();
        }
        }
        return Vec3(0.f);
    }

private:
    const MaterialLookup *m_materialLookupPtr;
    int m_materialID;
};

}
