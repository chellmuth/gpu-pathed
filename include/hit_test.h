#pragma once

#include "scene.h"
#include "scene_model.h"

namespace rays {

void hitTest(const Scene &scene, SceneModel &sceneModel, int pixelX, int pixelY);

}
