#include "renderers/g_buffer.h"

#include <cfloat>
#include <iostream>

#include "camera.h"
#include "frame.h"
#include "framebuffer.h"
#include "macro_helper.h"
#include "world.h"
#include "materials/lambertian.h"
#include "scene.h"
#include "core/vec3.h"

#define checkCudaErrors(result) { gpuAssert((result), __FILE__, __LINE__); }

namespace rays {

static constexpr bool debug = true;

GBuffer::GBuffer(BufferType bufferType)
    : m_bufferType(bufferType),
      m_currentSamples(0),
      m_shouldReset(false)
{}

__device__ static Vec3 calculateGBuffer(
    const Ray &ray,
    const World *world
) {
    HitRecord record;

    bool hit = world->hit(ray, 0.f, FLT_MAX, record);
    if (hit) {
        return (record.normal + 1.f) / 2.f;
    } else {
        return Vec3(0.f);
    }
}

__global__ static void renderKernel(
    Vec3 *passRadiances,
    int width, int height,
    World *world,
    Camera *camera
) {
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int col = threadIdx.x + blockIdx.x * blockDim.x;
    if ((row >= height) || (col >= width)) { return; }

    const int pixelIndex = row * width + col;

    const Ray cameraRay = camera->generateRay(row, col);
    const Vec3 Li = calculateGBuffer(cameraRay, world);

    passRadiances[pixelIndex] = Li;
}

void GBuffer::init(
    int width,
    int height,
    const Scene &scene
) {
    Renderer::init(width, height, scene);

    const int pixelCount = m_width * m_height;

    checkCudaErrors(cudaMalloc((void **)&dev_passRadiances, pixelCount * sizeof(Vec3)));
}

RenderRecord GBuffer::renderAsync(
    int spp,
    cudaGraphicsResource *pboResource,
    const Scene &scene,
    const CUDAGlobals &cudaGlobals
) {
    checkCudaErrors(cudaGraphicsMapResources(1, &pboResource, NULL));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dev_map, NULL, pboResource));

    const int blockWidth = 16;
    const int blockHeight = 16;

    const dim3 blocks(m_width / blockWidth + 1, m_height / blockHeight + 1);
    const dim3 threads(blockWidth, blockHeight);

    cudaEvent_t beginEvent, endEvent;
    cudaEventCreate(&beginEvent);
    cudaEventCreate(&endEvent);

    cudaEventRecord(beginEvent);

    if (m_shouldReset) {
        m_currentSamples = 0;
        m_shouldReset = false;
    }

    renderKernel<<<blocks, threads>>>(
        dev_passRadiances,
        m_width, m_height,
        cudaGlobals.d_world,
        cudaGlobals.d_camera
    );

    updateFramebuffer(
        dev_map,
        dev_passRadiances,
        dev_radiances,
        1,
        m_currentSamples,
        m_width,
        m_height,
        0
    );

    cudaEventRecord(endEvent);

    return RenderRecord{beginEvent, endEvent, 1};
}

bool GBuffer::pollRender(cudaGraphicsResource *pboResource, RenderRecord record)
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

void GBuffer::reset()
{
    m_shouldReset = true;
}

}
