#pragma once

#include "camera.h"
#include "materials/material_lookup.h"
#include "world.h"
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

    void mallocWorld(const SceneData &sceneData);
    void copySceneData(const SceneData &sceneData);

    MaterialLookup m_materialLookup;
    MaterialLookup *d_materialLookup;

    EnvironmentLight m_environmentLight;
    EnvironmentLight *d_environmentLight;

    Camera *d_camera;
    World *d_world;

    Triangle *d_triangles;
    Sphere *d_spheres;
    int *d_lightIndices;
};

}
