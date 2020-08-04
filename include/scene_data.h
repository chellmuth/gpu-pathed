#pragma once

#include <vector>

#include "materials/mirror.h"
#include "materials/material.h"
#include "materials/material_table.h"
#include "parsers/obj_parser.h"
#include "sphere.h"
#include "triangle.h"

namespace rays {

struct SceneData {
    std::vector<Triangle> triangles;
    std::vector<Sphere> spheres;
    std::vector<Material> materials;
    std::vector<Material> lambertians;
    std::vector<Mirror> mirrors;
    std::vector<int> lightIndices;

    bool isEmitter(MaterialIndex materialIndex) const {
        return !(lambertians[materialIndex.index].getEmit().isZero());
    }
};

}

namespace rays { namespace SceneAdapter {

struct ParseRequest {
    std::vector<ObjParser> objParsers;
    std::vector<Material> defaultMaterials;

    MaterialTable materialTable;
    std::vector<MaterialIndex> defaultMaterialIndices;
};

SceneData createSceneData(ParseRequest &request);
SceneData createSceneData(ObjParser &objParser);

} }
