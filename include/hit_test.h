#pragma once

#include "cuda_globals.h"
#include "scene_model.h"

namespace rays {

void hitTest(
    SceneModel &sceneModel,
    const CUDAGlobals &cudaGlobals,
    int pixelX,
    int pixelY,
    int width,
    int height
);

}
