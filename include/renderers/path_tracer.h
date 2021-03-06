#pragma once

#include <vector>

#include <curand_kernel.h>

#include "cuda_globals.h"
#include "renderers/renderer.h"
#include "render_record.h"
#include "scene.h"
#include "core/vec3.h"

namespace rays {

class PathTracer : public Renderer {
public:
    PathTracer();

    void init(int width, int height, const Scene &scene) override;
    RenderRecord renderAsync(
        int spp,
        cudaGraphicsResource *pboResource,
        const Scene &scene,
        const CUDAGlobals &cudaGlobals
    ) override;
    bool pollRender(cudaGraphicsResource *pboResource, RenderRecord record) override;

    void reset() override;
    int getSpp() const override { return m_currentSamples; }

private:
    int m_currentSamples;
    bool m_shouldReset;

    curandState *dev_randState;

    Vec3 *dev_passRadiances;
};

}
