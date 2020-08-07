#include "path_tracer.h"

#include <cfloat>
#include <iostream>

#include "camera.h"
#include "frame.h"
#include "framebuffer.h"
#include "macro_helper.h"
#include "primitive.h"
#include "materials/lambertian.h"
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

__device__ static Vec3 directSampleBSDF(
    const HitRecord &hitRecord,
    const BSDFSample &bsdfSample,
    const PrimitiveList *world,
    curandState &randState
) {
    if (!bsdfSample.isDelta) { return Vec3(0.f); }

    const Frame intersection(hitRecord.normal);
    const Vec3 bounceDirection = intersection.toWorld(bsdfSample.wiLocal);
    const Ray bounceRay(hitRecord.point, bounceDirection);

    HitRecord brdfRecord;
    const bool hit = world->hit(bounceRay, 1e-3, FLT_MAX, brdfRecord);
    if (!hit) { return Vec3(0.f); }
    const Vec3 emit = world->getEmit(brdfRecord.materialID, brdfRecord);
    if (emit.isZero()) { return Vec3(0.f); }

    const float brdfWeight = 1.f;

    const Vec3 brdfContribution = emit
        * brdfWeight
        * bsdfSample.f
        * bsdfSample.wiLocal.z()
        / bsdfSample.pdf;

    return brdfContribution;
}

__device__ static Vec3 directSampleLights(
    const HitRecord &hitRecord,
    const BSDFSample &bsdfSample,
    const PrimitiveList *world,
    curandState &randState
) {
    if (bsdfSample.isDelta) { return 0.f; }

    const LightSample lightSample = world->sampleDirectLights(hitRecord.point, randState);

    const Vec3 wiWorld = normalized(lightSample.point - hitRecord.point);
    const Ray shadowRay(hitRecord.point, wiWorld);

    HitRecord occlusionRecord;
    const bool occluded = world->hit(
        shadowRay,
        1e-4,
        (lightSample.point - hitRecord.point).length() - 2e-4,
        occlusionRecord
    );

    if (!occluded) {
        const Vec3 wi = Frame(hitRecord.normal).toLocal(wiWorld);
        const float pdf = lightSample.solidAnglePDF(hitRecord.point);
        const Vec3 emit = world->getEmit(lightSample.materialID);
        const Vec3 lightContribution = Vec3(1.f)
            * emit
            * world->f(hitRecord.materialID, hitRecord.wo, wi)
            * WorldFrame::absCosTheta(hitRecord.normal, wiWorld)
            / pdf;

        return lightContribution;
    } else {
        return Vec3(0.f);
    }
}

__device__ static Vec3 direct(
    const HitRecord &hitRecord,
    const BSDFSample &bsdfSample,
    const PrimitiveList *world,
    curandState &randState
) {
    Vec3 result(0.f);

    result += directSampleLights(hitRecord, bsdfSample, world, randState);
    result += directSampleBSDF(hitRecord, bsdfSample, world, randState);

    return result;
}

__device__ static Vec3 calculateLiNEE(
    const Ray& ray,
    const PrimitiveList *world,
    int maxDepth,
    curandState &randState
) {
    if (maxDepth == 0) { return Vec3(0.f); }

    Vec3 beta = Vec3(1.f);
    Vec3 result = Vec3(0.f);

    HitRecord record;
    bool hit = world->hit(ray, 0.f, FLT_MAX, record);
    if (hit) {
        const Vec3 emit = world->getEmit(record.materialID, record);

        if (!emit.isZero()) {
            result += emit * beta;
        }
    } else {
        return Vec3(0.f);
    }

    for (int path = 1; path < maxDepth; path++) {
        const Frame intersection(record.normal);

        const BSDFSample bsdfSample = world->sample(record.materialID, record, randState);
        result += direct(record, bsdfSample, world, randState) * beta;

        beta *= bsdfSample.f
            * intersection.absCosTheta(bsdfSample.wiLocal)
            / bsdfSample.pdf;

        const Vec3 bounceDirection = intersection.toWorld(bsdfSample.wiLocal);
        const Ray bounceRay(record.point, bounceDirection);
        hit = world->hit(bounceRay, 1e-3, FLT_MAX, record);
        if (!hit) {
            return result;
        }
    }

    return result;
}

__device__ static Vec3 calculateLiNaive(
    const Ray& ray,
    const PrimitiveList *world,
    int maxDepth,
    curandState &randState
) {
    if (maxDepth == 0) { return Vec3(0.f); }

    Vec3 beta = Vec3(1.f);
    Vec3 result = Vec3(0.f);

    HitRecord record;
    bool hit = world->hit(ray, 0.f, FLT_MAX, record);
    if (hit) {
       const Vec3 emit = world->getEmit(record.materialID, record);

        if (!emit.isZero()) {
            result += emit * beta;
        }
    } else {
        return Vec3(0.f);
    }

    for (int path = 1; path < maxDepth; path++) {
        const Frame intersection(record.normal);

        const BSDFSample bsdfSample = world->sample(record.materialID, record, randState);

        beta *= Vec3(1.f)
            * bsdfSample.f
            * intersection.absCosTheta(bsdfSample.wiLocal)
            / bsdfSample.pdf;

        const Vec3 bounceDirection = intersection.toWorld(bsdfSample.wiLocal);
        const Ray bounceRay(record.point, bounceDirection);
        hit = world->hit(bounceRay, 1e-3, FLT_MAX, record);
        if (hit) {
            const Vec3 emit = world->getEmit(record.materialID, record);
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
    int width, int height,
    curandState *randState,
    PrimitiveList *world,
    Camera *camera,
    int maxDepth,
    int spp,
    bool useNextEventEstimation
) {
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int col = threadIdx.x + blockIdx.x * blockDim.x;
    if ((row >= height) || (col >= width)) { return; }

    const int pixelIndex = row * width + col;
    curandState &localRand = randState[pixelIndex];
    for (int sample = 1; sample <= spp; sample++) {
        const Ray cameraRay = camera->generateRay(row, col, localRand);
        const Vec3 Li = useNextEventEstimation
            ? calculateLiNEE(cameraRay, world, maxDepth, localRand)
            : calculateLiNaive(cameraRay, world, maxDepth, localRand)
        ;

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
    Renderer::init(width, height, scene);

    const int pixelCount = m_width * m_height;

    checkCudaErrors(cudaMalloc((void **)&dev_randState, pixelCount * sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void **)&dev_passRadiances, pixelCount * sizeof(Vec3)));

    dim3 blocks(m_width, m_height);
    renderInit<<<blocks, 1>>>(m_width, m_height, dev_randState);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

RenderRecord PathTracer::renderAsync(
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
        dev_randState,
        cudaGlobals.d_world,
        cudaGlobals.d_camera,
        scene.getMaxDepth(),
        spp,
        scene.getNextEventEstimation()
    );

    updateFramebuffer(
        dev_map,
        dev_passRadiances,
        dev_radiances,
        spp,
        m_currentSamples,
        m_width,
        m_height,
        0
    );

    cudaEventRecord(endEvent);

    return RenderRecord{beginEvent, endEvent, spp};
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

    m_currentSamples += record.spp;
    return true;
}

void PathTracer::reset()
{
    m_shouldReset = true;
}

}
