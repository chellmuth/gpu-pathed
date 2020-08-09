#pragma once

#include "camera.h"
#include "materials/glass.h"
#include "materials/lambertian.h"
#include "materials/material_lookup.h"
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

    void initMaterials(const SceneData &sceneData);
    void updateMaterials(const SceneData &sceneData);

    void copySceneData(const SceneData &sceneData);
    void mallocWorld(const SceneData &sceneData);

    MaterialLookup m_materialLookup;

    Camera *d_camera;
    PrimitiveList *d_world;

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
