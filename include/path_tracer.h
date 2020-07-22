#pragma once

#include <memory>

#include <curand_kernel.h>
#include <cuda_gl_interop.h>

#include "cuda_globals.h"
#include "vec3.h"

namespace rays {

class PathTracer {
public:
    PathTracer();

    PathTracer(const PathTracer &other) = delete;
    PathTracer(PathTracer&& other) = delete;

    void init(GLuint pbo, int width, int height);
    void render(const CUDAGlobals &cudaGlobals);

    void reset();
    int getSpp() const { return m_currentSamples; }

private:
    int m_width, m_height;
    int m_currentSamples;

    cudaGraphicsResource *m_cudaPbo;
    curandState *dev_randState;

    Vec3 *dev_radiances;
    uchar4 *dev_map;
};

}
