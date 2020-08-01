#pragma once

#include <vector>

#include "material.h"
#include "parsers/obj_parser.h"
#include "sphere.h"
#include "triangle.h"

namespace rays {

struct SceneData {
    std::vector<Triangle> triangles;
    std::vector<Sphere> spheres;
    std::vector<Material> materials;
    std::vector<int> lightIndices;
};

}

namespace rays { namespace SceneAdapter {

struct ParseRequest {
    std::vector<ObjParser> objParsers;
    std::vector<Material> defaultMaterials;
};

SceneData createSceneData(ParseRequest &request);
SceneData createSceneData(ObjParser &objParser);

} }
