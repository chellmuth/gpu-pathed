#pragma once

#include <cuda_gl_interop.h>

#include "cuda_globals.h"
#include "render_record.h"
#include "scene.h"
#include "vec3.h"

namespace rays {

class Renderer {
public:
    virtual void init(int width, int height, const Scene &scene) = 0;
    virtual RenderRecord renderAsync(cudaGraphicsResource *pboResource, const CUDAGlobals &cudaGlobals) = 0;
    virtual bool pollRender(cudaGraphicsResource *pboResource, RenderRecord record) = 0;

    virtual void reset() = 0;
    virtual int getSpp() const = 0;
};

}
