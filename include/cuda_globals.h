#pragma once

#include "camera.h"
#include "primitive.h"
#include "sphere.h"
#include "triangle.h"

namespace rays {

class CUDAGlobals {
public:
    void copyCamera(const Camera &camera);
    void mallocWorld();

    Camera *d_camera;
    PrimitiveList *d_world;

    Material *d_materials;
    Triangle *d_triangles;
    Sphere *d_spheres;
};

}
