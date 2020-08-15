#pragma once

#include <cuda_gl_interop.h>

#include "cuda_globals.h"
#include "render_record.h"
#include "renderers/renderer.h"
#include "scene.h"
#include "core/vec3.h"

namespace rays {

enum class BufferType {
    Normals
};

class GBuffer : public Renderer {
public:
    GBuffer(BufferType bufferType);

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

    Vec3 *dev_passRadiances;

    BufferType m_bufferType;
};

}
