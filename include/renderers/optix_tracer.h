#pragma once

#include <memory>
#include <vector>

#include <cuda_gl_interop.h>
#include <curand_kernel.h>

#include "cuda_globals.h"
#include "render_record.h"
#include "renderers/optix.h"
#include "renderers/renderer.h"
#include "scene.h"
#include "core/vec3.h"

namespace rays {

class OptixTracer : public Renderer {
public:
    OptixTracer();

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

    Optix m_optix;
};

}
