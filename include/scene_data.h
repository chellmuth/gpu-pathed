#pragma once

#include <vector>

#include "materials/mirror.h"
#include "materials/material.h"
#include "materials/material_store.h"
#include "materials/material_table.h"
#include "parsers/obj_parser.h"
#include "sphere.h"
#include "triangle.h"

namespace rays {

struct SceneData {
    std::vector<Triangle> triangles;
    std::vector<Sphere> spheres;
    MaterialStore materialStore;
    std::vector<int> lightIndices;

    bool isEmitter(MaterialIndex materialIndex) const {
        switch (materialIndex.materialType) {
        case MaterialType::Lambertian: {
            return !materialStore.getLambertians()[materialIndex.index].getEmit().isZero();
        }
        case MaterialType::Mirror: {
            return !materialStore.getMirrors()[materialIndex.index].getEmit().isZero();
        }
        }
        return false;
    }
};

}

namespace rays { namespace SceneAdapter {

struct ParseRequest {
    std::vector<ObjParser> objParsers;
    MaterialStore materialStore;
    std::vector<int> defaultMaterialIDs;
};

SceneData createSceneData(ParseRequest &request);
SceneData createSceneData(ObjParser &objParser);

} }
