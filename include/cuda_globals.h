#pragma once

#include "camera.h"
#include "materials/glass.h"
#include "materials/lambertian.h"
#include "materials/material_table.h"
#include "materials/mirror.h"
#include "materials/types.h"
#include "primitive.h"
#include "scene_data.h"
#include "sphere.h"
#include "triangle.h"

namespace rays {

class CUDAGlobals {
public:
    void mallocCamera();
    void copyCamera(const Camera &camera);

    void mallocMaterials(const SceneData &sceneData);
    void copyMaterials(const SceneData &sceneData);
    void freeMaterials();

    void copySceneData(const SceneData &sceneData);
    void mallocWorld(const SceneData &sceneData);

    Camera *d_camera;
    PrimitiveList *d_world;

    Lambertian *d_materials;

    MaterialLookup *d_materialLookup;
    MaterialIndex *d_materialIndices;
    Lambertian *d_lambertians;
    Mirror *d_mirrors;
    Glass *d_glasses;

    Triangle *d_triangles;
    Sphere *d_spheres;
    int *d_lightIndices;
};

}
