#include "path_tracer.h"

#include <cfloat>
#include <iostream>

#include "camera.h"
#include "frame.h"
#include "framebuffer.h"
#include "macro_helper.h"
#include "primitive.h"
#include "material.h"
#include "scene.h"
#include "vec3.h"

#define checkCudaErrors(result) { gpuAssert((result), __FILE__, __LINE__); }

namespace rays {

static constexpr bool debug = true;

PathTracer::PathTracer()
    : m_currentSamples(0),
      m_shouldReset(false)
{}

__global__ static void renderInit(int width, int height, curandState *randState)
{
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int col = threadIdx.x + blockIdx.x * blockDim.x;

    if ((row >= height) || (col >= width)) return;
    const int pixelIndex = row * width + col;

    constexpr int seed = 0;
    curand_init(seed, pixelIndex, 0, &randState[pixelIndex]);
}

__device__ static Vec3 calculateLi(const Ray& ray, const PrimitiveList *world, curandState &randState)
{
    Vec3 beta = Vec3(1.f);
    Vec3 result = Vec3(0.f);

    HitRecord record;
    bool hit = world->hit(ray, 0.f, FLT_MAX, record);
    if (hit) {
        const Material &emitMaterial = world->getMaterial(record.materialIndex);
        const Vec3 emit = emitMaterial.getEmit(record);

        if (!emit.isZero()) {
            result += emit * beta;
        }
    } else {
        return Vec3(0.f);
    }

    for (int path = 2; path < 3; path++) {
        const Frame intersection(record.normal);
        const Material &material = world->getMaterial(record.materialIndex);
        float pdf;

        const Vec3 wi = material.sample(record, &pdf, randState);
        const Vec3 bounceDirection = intersection.toWorld(wi);

        beta *= material.f(record.wo, wi) * intersection.cosTheta(wi) / pdf;

        const Ray bounceRay(record.point, bounceDirection);
        hit = world->hit(bounceRay, 1e-3, FLT_MAX, record);
        if (hit) {
            const Material &emitMaterial = world->getMaterial(record.materialIndex);
            const Vec3 emit = emitMaterial.getEmit(record);
            if (!emit.isZero()) {
                result += emit * beta;
            }
        } else {
            return result;
        }
    }

    return result;
}

__global__ static void renderKernel(
    Vec3 *passRadiances,
    Camera *camera,
    int spp,
    int width, int height,
    PrimitiveList *world,
    curandState *randState
) {
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int col = threadIdx.x + blockIdx.x * blockDim.x;
    if ((row >= height) || (col >= width)) { return; }

    const int pixelIndex = row * width + col;
    curandState &localRand = randState[pixelIndex];
    for (int sample = 1; sample <= spp; sample++) {
        const Ray cameraRay = camera->generateRay(row, col, localRand);
        const Vec3 Li = calculateLi(cameraRay, world, localRand);

        if (sample == 1) {
            passRadiances[pixelIndex] = Li;
        } else {
            passRadiances[pixelIndex] += Li / spp;
        }
    }
}

void PathTracer::init(
    int width,
    int height,
    const Scene &scene
) {
    m_width = width;
    m_height = height;
    const int pixelCount = m_width * m_height;

    checkCudaErrors(cudaMalloc((void **)&dev_randState, pixelCount * sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void **)&dev_radiances, pixelCount * sizeof(Vec3)));
    checkCudaErrors(cudaMalloc((void **)&dev_passRadiances, pixelCount * sizeof(Vec3)));

    dim3 blocks(m_width, m_height);
    renderInit<<<blocks, 1>>>(m_width, m_height, dev_randState);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

static constexpr int samplesPerPass = 1;
RenderRecord PathTracer::renderAsync(cudaGraphicsResource *pboResource, const CUDAGlobals &cudaGlobals)
{
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
        cudaGlobals.d_camera,
        samplesPerPass,
        m_width, m_height,
        cudaGlobals.d_world,
        dev_randState
    );

    updateFramebuffer(
        dev_map,
        dev_passRadiances,
        dev_radiances,
        samplesPerPass,
        m_currentSamples,
        m_width,
        m_height,
        0
    );

    cudaEventRecord(endEvent);

    return RenderRecord{beginEvent, endEvent};
}

bool PathTracer::pollRender(cudaGraphicsResource *pboResource, RenderRecord record)
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

    m_currentSamples += samplesPerPass;
    return true;
}

void PathTracer::reset()
{
    m_shouldReset = true;
}

}
