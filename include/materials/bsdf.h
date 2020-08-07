#pragma once

#include "intersection.h"
#include "materials/bsdf_sample.h"
#include "materials/material_table.h"
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
        }
        return Vec3(0.f);
    }

    __device__ BSDFSample sample(
        const Intersection &intersection,
        unsigned int &seed
    ) {
        const MaterialIndex index = m_materialLookupPtr->indices[m_materialID];
        switch(index.materialType) {
        case MaterialType::Lambertian: {
            return m_materialLookupPtr->lambertians[index.index].sample(intersection.woLocal, seed);
        }
        case MaterialType::Mirror: {
            return m_materialLookupPtr->mirrors[index.index].sample(intersection.woLocal);
        }
        case MaterialType::Glass: {
            const float xi1 = rnd(seed);
            return m_materialLookupPtr->glasses[index.index].sample(intersection.woLocal, xi1);
        }
        }
        return {};
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
        }
        return Vec3(0.f);
    }

private:
    const MaterialLookup *m_materialLookupPtr;
    int m_materialID;
};

}
