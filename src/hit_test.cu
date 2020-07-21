#include "hit_test.h"

#include <cfloat>
#include <iostream>
#include <memory>

#include "camera.h"
#include "material.h"
#include "primitive.h"
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

__global__ static void hitTestKernel(
    int width,
    int height,
    int pixelX,
    int pixelY,
    PrimitiveList **world,
    bool *isHit,
    int *materialIndex
) {
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int col = threadIdx.x + blockIdx.x * blockDim.x;

    if ((col != pixelX) || (row != pixelY)) { return; }

    const Camera camera(
        Vec3(0.f, 0.3f, 5.f),
        30.f / 180.f * M_PI,
        { width, height }
    );

    curandState_t randState;
    curand_init(0, 0, 0, &randState);
    const Ray cameraRay = camera.generateRay(row, col, randState);
    HitRecord record;

    bool hit = (*world)->hit(cameraRay, 0.f, FLT_MAX, record);
    if (hit) {
        *materialIndex = record.materialIndex;
    }

    *isHit = hit;
}

void hitTest(const Scene &scene, SceneModel &sceneModel, int pixelX, int pixelY)
{
    std::cout << "Testing: " << pixelX << " " << pixelY << std::endl;

    Primitive **dev_primitives;
    Material *dev_materials;
    PrimitiveList **dev_world;

    bool *dev_isHit;
    int *dev_materialIndex;

    checkCudaErrors(cudaMalloc((void **)&dev_primitives, primitiveCount * sizeof(Primitive *)));
    checkCudaErrors(cudaMalloc((void **)&dev_materials, materialCount * sizeof(Material)));
    checkCudaErrors(cudaMalloc((void **)&dev_world, sizeof(PrimitiveList *)));
    checkCudaErrors(cudaMalloc((void **)&dev_isHit, sizeof(bool)));
    checkCudaErrors(cudaMalloc((void **)&dev_materialIndex, sizeof(int)));

    createWorld<<<1, 1>>>(
        dev_primitives,
        dev_materials,
        dev_world,
        sceneModel.getLightPosition(),
        false
    );

    checkCudaErrors(cudaGetLastError());

    constexpr int width = 640;
    constexpr int height = 360;
    dim3 blocks(width, height);
    hitTestKernel<<<blocks, 1>>>(
        width,
        height,
        pixelX,
        pixelY,
        dev_world,
        dev_isHit,
        dev_materialIndex
    );

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    bool isHit;
    int materialIndex;
    checkCudaErrors(cudaMemcpy(&isHit, dev_isHit, sizeof(bool), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&materialIndex, dev_materialIndex, sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "IS HIT?? " << isHit << std::endl;
    if (isHit) {
        std::cout << "MATERIAL: " << materialIndex << std::endl;

        const Material &material = scene.getMaterial(materialIndex);
        sceneModel.setMaterialIndex(materialIndex, material.getAlbedo());
    } else {
        sceneModel.setMaterialIndex(-1, Vec3(0.f));
    }


}

}

#undef checkCudaErrors
