#pragma once

#include "camera.h"

namespace rays {

class CUDAGlobals {
public:
    void copyCamera(const Camera &camera);

    Camera *d_camera;
};

}
