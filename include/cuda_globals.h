#pragma once

#include "camera.h"
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
    Triangle *d_triangles;
    Sphere *d_spheres;
};

}
