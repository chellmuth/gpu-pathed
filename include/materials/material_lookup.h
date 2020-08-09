#pragma once

#include <vector>

#include "materials/glass.h"
#include "materials/lambertian.h"
#include "materials/mirror.h"
#include "materials/types.h"
#include "scene_data.h"

namespace rays {

struct MaterialLookup {
    MaterialIndex *indices;

    Lambertian *lambertians;
    Mirror *mirrors;
    Glass *glasses;

    void mallocMaterials(const SceneData &sceneData);
    void copyMaterials(const SceneData &sceneData);
    void freeMaterials();
};

}
