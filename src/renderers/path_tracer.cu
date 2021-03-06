#include "renderers/path_tracer.h"

#include <cfloat>
#include <iostream>

#include "core/camera.h"
#include "core/vec3.h"
#include "frame.h"
#include "framebuffer.h"
#include "macro_helper.h"
#include "math/mis.h"
#include "scene.h"
#include "world.h"

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
    const World *world,
    curandState &randState
) {
    if (bsdfSample.f.isZero()) { return Vec3(0.f); }

    const Frame intersection(hitRecord.normal);
    const Vec3 bounceDirection = intersection.toWorld(bsdfSample.wiLocal);
    const Ray bounceRay(hitRecord.point, bounceDirection);

    HitRecord brdfRecord;
    const bool hit = world->hit(bounceRay, 1e-4, FLT_MAX, brdfRecord);
    if (hit) {
        const Vec3 emit = world->getEmit(brdfRecord.materialID, brdfRecord);
        if (emit.isZero()) { return Vec3(0.f); }

        const float lightPDF = world->pdfSceneLights(hitRecord.point, brdfRecord);

        const float bsdfWeight = bsdfSample.isDelta
            ? 1.f
            : rays::MIS::balanceWeight(1, 1, bsdfSample.pdf, lightPDF);

        const Vec3 brdfContribution = emit
            * bsdfWeight
            * bsdfSample.f
            * TangentFrame::absCosTheta(bsdfSample.wiLocal)
            / bsdfSample.pdf;
        return brdfContribution;
    } else {
        const float lightPDF = world->pdfEnvironmentLight(bounceDirection);

        const float bsdfWeight = bsdfSample.isDelta
            ? 1.f
            : rays::MIS::balanceWeight(1, 1, bsdfSample.pdf, lightPDF);

        return world->environmentL(bounceDirection)
            * bsdfWeight
            * bsdfSample.f
            * TangentFrame::absCosTheta(bsdfSample.wiLocal)
            / bsdfSample.pdf;
    }
}

__device__ static Vec3 directSampleLights(
    const HitRecord &hitRecord,
    const BSDFSample &bsdfSample,
    const World *world,
    curandState &randState
) {
    if (bsdfSample.isDelta) { return 0.f; }

    const LightSample lightSample = world->sampleDirectLights(
        hitRecord.point,
        Frame(hitRecord.normal),
        randState
    );

    if (dot(lightSample.normal, lightSample.wi) >= 0.f) {
        return rays::Vec3(0.f);
    }

    const Ray shadowRay(hitRecord.point, lightSample.wi);

    HitRecord occlusionRecord;
    const bool occluded = world->hit(
        shadowRay,
        1e-4,
        lightSample.distance - 2e-4,
        occlusionRecord
    );

    if (occluded) {
        return Vec3(0.f);
    }

    const Vec3 wiLocal = Frame(hitRecord.normal).toLocal(lightSample.wi);
    const float bsdfPDF = world->pdfBSDF(bsdfSample.materialID, hitRecord.wo, wiLocal);
    const float lightWeight = rays::MIS::balanceWeight(1, 1, lightSample.pdf, bsdfPDF);

    const Vec3 lightContribution = Vec3(1.f)
        * lightSample.emitted
        * lightWeight
        * world->f(hitRecord.materialID, hitRecord.wo, wiLocal)
        * WorldFrame::absCosTheta(hitRecord.normal, lightSample.wi)
        / lightSample.pdf;
    return lightContribution;
}

__device__ static Vec3 direct(
    const HitRecord &hitRecord,
    const BSDFSample &bsdfSample,
    const World *world,
    curandState &randState
) {
    Vec3 result(0.f);

    result += directSampleLights(hitRecord, bsdfSample, world, randState);
    result += directSampleBSDF(hitRecord, bsdfSample, world, randState);

    return result;
}

__device__ static Vec3 calculateLiNEE(
    const Ray& ray,
    const World *world,
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
        return world->environmentL(ray.direction());
    }

    for (int path = 1; path < maxDepth; path++) {
        const Frame intersection(record.normal);

        const BSDFSample bsdfSample = world->sample(record.materialID, record, randState);
        result += beta * direct(record, bsdfSample, world, randState);

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
    const World *world,
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
        return world->environmentL(ray.direction());
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
            return result + world->environmentL(bounceRay.direction()) * beta;
        }
    }

    return result;
}

__global__ static void renderKernel(
    Vec3 *passRadiances,
    int width, int height,
    curandState *randState,
    World *world,
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
            passRadiances[pixelIndex] = Li / spp;
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
