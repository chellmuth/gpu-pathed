#include "renderers/optix_tracer.h"

#include <cfloat>
#include <iostream>

#include "camera.h"
#include "color.h"
#include "frame.h"
#include "macro_helper.h"
#include "renderers/optix.h"
#include "primitive.h"
#include "material.h"
#include "scene.h"
#include "vec3.h"

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
    m_width = width;
    m_height = height;
    const int pixelCount = m_width * m_height;

    m_optix.init(m_width, m_height, scene);

    checkCudaErrors(cudaMalloc((void **)&dev_radiances, pixelCount * sizeof(Vec3)));
}

RenderRecord OptixTracer::renderAsync(
    cudaGraphicsResource *pboResource,
    const Scene &scene,
    const CUDAGlobals &cudaGlobals
) {
    checkCudaErrors(cudaGraphicsMapResources(1, &pboResource, NULL));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dev_map, NULL, pboResource));

    if (m_shouldReset) {
        m_currentSamples = 0;
        m_optix.updateCamera(scene);
        m_optix.updateMaterials(scene);
        m_shouldReset = false;
    }

    cudaEvent_t beginEvent, endEvent;
    cudaEventCreate(&beginEvent);
    cudaEventCreate(&endEvent);
    cudaEventRecord(beginEvent);

    uchar4 *image = m_optix.launch(m_currentSamples);
    checkCudaErrors(cudaMemcpy(dev_map, image, m_width * m_height * sizeof(uchar4), cudaMemcpyHostToDevice));

    cudaEventRecord(endEvent);

    free(image);
    return RenderRecord{beginEvent, endEvent};
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

    m_currentSamples += 1;
    return true;
}

void OptixTracer::reset()
{
    m_shouldReset = true;
}

}
