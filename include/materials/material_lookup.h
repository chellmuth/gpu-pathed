#pragma once

#include <vector>

#include "core/vec3.h"
#include "materials/glass.h"
#include "materials/lambertian.h"
#include "materials/microfacet.h"
#include "materials/mirror.h"
#include "materials/types.h"
#include "scene_data.h"

namespace rays {

struct MaterialLookup {
    MaterialIndex *indices;

    Lambertian *lambertians;
    Mirror *mirrors;
    Glass *glasses;
    Microfacet *microfacets;

    __device__ Vec3 getEmit(const int materialID) const {
        MaterialIndex index = indices[materialID];
        return getEmit(index);
    }

    void mallocMaterials(const SceneData &sceneData);
    void copyMaterials(const SceneData &sceneData);
    void freeMaterials();

private:

    __device__ Vec3 getEmit(const MaterialIndex index) const {
        switch(index.materialType) {
        case MaterialType::Lambertian: {
            return lambertians[index.index].getEmit();
        }
        case MaterialType::Mirror: {
            return mirrors[index.index].getEmit();
        }
        case MaterialType::Glass: {
            return glasses[index.index].getEmit();
        }
        case MaterialType::Microfacet: {
            return microfacets[index.index].getEmit();
        }
        }
        return Vec3(0.f);
    }
};

}
