#pragma once

#include <memory>

#include <curand_kernel.h>
#include <cuda_gl_interop.h>

#include "cuda_globals.h"
#include "vec3.h"

namespace rays {

struct RenderRecord {
    cudaEvent_t beginEvent;
    cudaEvent_t endEvent;
};

class PathTracer {
public:
    PathTracer();

    PathTracer(const PathTracer &other) = delete;
    PathTracer(PathTracer&& other) = delete;

    void init(int width, int height);
    RenderRecord renderAsync(cudaGraphicsResource *pboResource, const CUDAGlobals &cudaGlobals);
    bool pollRender(cudaGraphicsResource *pboResource, RenderRecord record);

    void reset();
    int getSpp() const { return m_currentSamples; }

private:
    int m_width, m_height;
    int m_currentSamples;
    bool m_shouldReset;

    curandState *dev_randState;

    Vec3 *dev_radiances;
    uchar4 *dev_map;
};

}
