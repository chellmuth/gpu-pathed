#pragma once

#include <memory>

#include <curand_kernel.h>
#include <cuda_gl_interop.h>

#include "cuda_globals.h"
#include "hit_test.h"
#include "primitive.h"
#include "scene_model.h"
#include "scene.h"
#include "vec3.h"

namespace rays {

class PathTracer {
public:
    PathTracer();

    PathTracer(const PathTracer &other) = delete;
    PathTracer(PathTracer&& other) = delete;

    void init(GLuint pbo, int width, int height);
    void render();

    SceneModel& getSceneModel();

    void test(int x, int y) {
        hitTest(*m_scene, *m_sceneModel, *m_cudaGlobals, x, y);
    }

private:
    std::unique_ptr<CUDAGlobals> m_cudaGlobals;
    std::unique_ptr<Scene> m_scene;
    std::unique_ptr<SceneModel> m_sceneModel;

    int m_width, m_height;
    int m_currentSamples;

    cudaGraphicsResource *m_cudaPbo;

    curandState *dev_randState;
    Primitive **dev_primitives;
    Material *dev_materials;
    PrimitiveList **dev_world;

    Vec3 *dev_radiances;
    uchar4 *dev_map;
};

}
