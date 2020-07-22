#pragma once

#include "camera.h"
#include "primitive.h"

namespace rays {

class CUDAGlobals {
public:
    void copyCamera(const Camera &camera);
    void mallocWorld();

    Camera *d_camera;
    PrimitiveList *d_world;
};

}
