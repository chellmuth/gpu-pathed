#pragma once

#include <memory>

#include <curand_kernel.h>
#include <cuda_gl_interop.h>

#include "primitive.h"
#include "scene_model.h"
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

private:
    std::unique_ptr<SceneModel> m_sceneModel;

    int m_width, m_height;
    int m_currentSamples;

    cudaGraphicsResource *m_cudaPbo;

    curandState *dev_randState;
    Primitive **dev_primitives;
    PrimitiveList **dev_world;

    Vec3 *dev_radiances;
    uchar4 *dev_map;
};

}
