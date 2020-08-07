#pragma once

#include <vector>

#include <cuda_gl_interop.h>

#include "cuda_globals.h"
#include "render_record.h"
#include "scene.h"
#include "vec3.h"

namespace rays {

class Renderer {
public:
    virtual void init(int width, int height, const Scene &scene);
    virtual RenderRecord renderAsync(
        int spp,
        cudaGraphicsResource *pboResource,
        const Scene &scene,
        const CUDAGlobals &cudaGlobals
    ) = 0;
    virtual bool pollRender(cudaGraphicsResource *pboResource, RenderRecord record) = 0;

    virtual void reset() = 0;
    virtual int getSpp() const = 0;

    virtual std::vector<float> getRadianceBuffer() const;

protected:
    int m_width, m_height;

    Vec3 *dev_radiances;
    uchar4 *dev_map;
};

}
