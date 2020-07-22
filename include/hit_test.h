#pragma once

#include "cuda_globals.h"
#include "scene.h"
#include "scene_model.h"

namespace rays {

void hitTest(
    const Scene &scene,
    SceneModel &sceneModel,
    const CUDAGlobals &cudaGlobals,
    int pixelX,
    int pixelY
);

}
