#include "path_tracer.h"

#include <cfloat>
#include <iostream>

#include "camera.h"
#include "frame.h"
#include "primitive.h"
#include "material.h"
#include "scene.h"
#include "vec3.h"

#define checkCudaErrors(result) { gpuAssert((result), __FILE__, __LINE__); }
static void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess) {
		fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

namespace rays {

static constexpr bool debug = false;

PathTracer::PathTracer()
    : m_currentSamples(0)
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
        const Vec3 direction = normalized(ray.direction());
        const float t = 0.5f * (direction.y() + 1.0f);
        return Vec3(1.f - t) + t * Vec3(0.5f, 0.7f, 1.f);
    }

    for (int path = 2; path < 10; path++) {
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
            const Vec3 direction = normalized(ray.direction());
            const float t = 0.5f * (direction.y() + 1.f);
            const Vec3 skyRadiance = (Vec3(1.f - t) + t * Vec3(0.5f, 0.7f, 1.f)) * 0.5f;
            return result + skyRadiance * beta;
        }
    }

    return result;
}

__global__ static void renderKernel(
    uchar4 *fb,
    Vec3 *radiances,
    Camera *camera,
    int spp,
    int currentSamples,
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

        const int spp = currentSamples + sample;

        Vec3 next;
        if (spp > 1) {
            const Vec3 current = radiances[pixelIndex];
            next = current * (spp - 1) / spp + (Li / spp);
        } else {
            next = Li;
        }

        radiances[pixelIndex] = next;
    }

    const Vec3 finalRadiance = radiances[pixelIndex];

    fb[pixelIndex].x = fmaxf(0.f, fminf(1.f, finalRadiance.x())) * 255;
    fb[pixelIndex].y = fmaxf(0.f, fminf(1.f, finalRadiance.y())) * 255;
    fb[pixelIndex].z = fmaxf(0.f, fminf(1.f, finalRadiance.z())) * 255;
    fb[pixelIndex].w = 255;
}

void PathTracer::init(
    GLuint pbo,
    int width,
    int height
) {
    checkCudaErrors(
        cudaGraphicsGLRegisterBuffer(
            &m_cudaPbo,
            pbo,
            cudaGraphicsRegisterFlagsWriteDiscard
        )
    );

    m_width = width;
    m_height = height;
    const int pixelCount = m_width * m_height;

    checkCudaErrors(cudaMalloc((void **)&dev_randState, pixelCount * sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void **)&dev_radiances, pixelCount * sizeof(Vec3)));

    dim3 blocks(m_width, m_height);
    renderInit<<<blocks, 1>>>(m_width, m_height, dev_randState);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void PathTracer::render(const CUDAGlobals &cudaGlobals)
{
    checkCudaErrors(cudaGraphicsMapResources(1, &m_cudaPbo, NULL));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dev_map, NULL, m_cudaPbo));

    const int samplesPerPass = 1;

    const int blockWidth = 16;
    const int blockHeight = 16;

    const dim3 blocks(m_width / blockWidth + 1, m_height / blockHeight + 1);
    const dim3 threads(blockWidth, blockHeight);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    renderKernel<<<blocks, threads>>>(
        dev_map,
        dev_radiances,
        cudaGlobals.d_camera,
        samplesPerPass,
        m_currentSamples,
        m_width, m_height,
        cudaGlobals.d_world,
        dev_randState
    );

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    if (debug) {
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "CUDA Frame: " << milliseconds << std::endl;
    }

    m_currentSamples += samplesPerPass;

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cudaPbo, NULL));
}

void PathTracer::reset() 
{
    m_currentSamples = 0;
}

}

#undef checkCudaErrors
