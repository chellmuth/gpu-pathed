#include "renderers/optix_tracer.h"

#include <cfloat>
#include <iostream>

#include "core/camera.h"
#include "core/color.h"
#include "frame.h"
#include "macro_helper.h"
#include "renderers/optix.h"
#include "world.h"
#include "materials/lambertian.h"
#include "scene.h"
#include "core/vec3.h"

#define checkCudaErrors(result) { gpuAssert((result), __FILE__, __LINE__); }

namespace rays {

static constexpr bool debug = true;

OptixTracer::OptixTracer()
    : m_currentSamples(0),
      m_shouldReset(false)
{}

void OptixTracer::init(
    int width,
    int height,
    const Scene &scene
) {
    Renderer::init(width, height, scene);

    m_optix.init(m_width, m_height, scene);
}

RenderRecord OptixTracer::renderAsync(
    int spp,
    cudaGraphicsResource *pboResource,
    const Scene &scene,
    const CUDAGlobals &cudaGlobals
) {
    checkCudaErrors(cudaGraphicsMapResources(1, &pboResource, NULL));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dev_map, NULL, pboResource));

    if (m_shouldReset) {
        m_currentSamples = 0;

        m_optix.updateMaxDepth(scene);
        m_optix.updateNextEventEstimation(scene);
        m_optix.updateCamera(scene);
        m_optix.updateMaterials(scene);

        m_shouldReset = false;
    }

    cudaEvent_t beginEvent, endEvent;
    cudaEventCreate(&beginEvent);
    cudaEventCreate(&endEvent);
    cudaEventRecord(beginEvent);

    uchar4 *image = m_optix.launch(dev_radiances, spp, m_currentSamples);
    checkCudaErrors(cudaMemcpy(dev_map, image, m_width * m_height * sizeof(uchar4), cudaMemcpyHostToDevice));

    cudaEventRecord(endEvent);

    free(image);
    return RenderRecord{beginEvent, endEvent, spp};
}

bool OptixTracer::pollRender(cudaGraphicsResource *pboResource, RenderRecord record)
{
    if (cudaEventQuery(record.endEvent) != cudaSuccess) {
        return false;
    }

    cudaEventSynchronize(record.endEvent);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGraphicsUnmapResources(1, &pboResource, NULL));

    if (debug) {
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, record.beginEvent, record.endEvent);
        std::cout << "CUDA Frame: " << milliseconds << "ms" << std::endl;
    }

    m_currentSamples += record.spp;
    return true;
}

void OptixTracer::reset()
{
    m_shouldReset = true;
}

}
