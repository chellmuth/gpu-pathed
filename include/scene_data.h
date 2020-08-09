#pragma once

#include <memory>
#include <vector>

#include "materials/glass.h"
#include "materials/lambertian.h"
#include "materials/material_store.h"
#include "materials/material_table.h"
#include "materials/mirror.h"
#include "materials/params.h"
#include "parsers/obj_parser.h"
#include "sphere.h"
#include "triangle.h"

namespace rays {

struct SceneData {
    std::vector<Triangle> triangles;
    std::vector<Sphere> spheres;
    std::vector<int> lightIndices;

    std::vector<std::unique_ptr<MaterialParams> > materialParams;

    SceneData() = default;
    SceneData(SceneData&&) = default;
    SceneData& operator=(SceneData&&) = default;
    SceneData(const SceneData&) = delete;
    SceneData& operator=(const SceneData&) = delete;
    ~SceneData() = default;

    int materialIDsCount() const {
        return materialParams.size();
    }

    bool isEmitter(int materialID) const {
        return !materialParams[materialID]->getEmit().isZero();
    }
};

}

namespace rays { namespace SceneAdapter {

struct ParseRequest {
    std::vector<ObjParser> objParsers;
    std::vector<std::unique_ptr<MaterialParams> > materialParams;
    std::vector<int> defaultMaterialParamsIDs;
};

SceneData createSceneData(ParseRequest &request);
SceneData createSceneData(ObjParser &objParser);

} }
