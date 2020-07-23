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
};

}

namespace rays { namespace SceneAdapter {

SceneData createSceneData(ObjParser &objParser);
SceneData createSceneData(float lightPosition);

} }
