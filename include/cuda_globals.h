#pragma once

#include "camera.h"
#include "materials/dummy.h"
#include "materials/material.h"
#include "primitive.h"
#include "scene_data.h"
#include "sphere.h"
#include "triangle.h"

namespace rays {

class CUDAGlobals {
public:
    void mallocCamera();
    void copyCamera(const Camera &camera);

    void copySceneData(const SceneData &sceneData);
    void mallocWorld(const SceneData &sceneData);

    Camera *d_camera;
    PrimitiveList *d_world;

    Material *d_materials;

    Material *d_lambertians;
    Dummy *d_dummies;

    Triangle *d_triangles;
    Sphere *d_spheres;
    int *d_lightIndices;
};

}
