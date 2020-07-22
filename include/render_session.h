#pragma once

#include "cuda_globals.h"
#include "hit_test.h"
#include "material.h"
#include "path_tracer.h"
#include "primitive.h"
#include "scene.h"
#include "scene_model.h"

namespace rays {

class RenderSession {
public:
    RenderSession();

    RenderSession(const RenderSession &other) = delete;
    RenderSession(RenderSession&& other) = delete;

    SceneModel& getSceneModel();

    void init(GLuint pbo, int width, int height);

    void hitTest(int x, int y) {
        rays::hitTest(*m_sceneModel, *m_cudaGlobals, x, y, m_width, m_height);
    }

    void render() { m_pathTracer->render(*m_cudaGlobals); }

private:
    std::unique_ptr<PathTracer> m_pathTracer;
    int m_width, m_height;

    std::unique_ptr<CUDAGlobals> m_cudaGlobals;
    std::unique_ptr<Scene> m_scene;
    std::unique_ptr<SceneModel> m_sceneModel;

    Primitive **dev_primitives;
    Material *dev_materials;
};

}
