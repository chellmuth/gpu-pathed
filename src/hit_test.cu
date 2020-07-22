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


struct HitTest {
    int materialIndex;
    bool isHit;
};

__global__ static void hitTestKernel(
    int pixelX,
    int pixelY,
    PrimitiveList **world,
    Camera *camera,
    HitTest *hitTest
) {
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int col = threadIdx.x + blockIdx.x * blockDim.x;

    if ((col != pixelX) || (row != pixelY)) { return; }

    const Ray cameraRay = camera->generateRay(row, col);
    HitRecord record;

    bool isHit = (*world)->hit(cameraRay, 0.f, FLT_MAX, record);
    if (isHit) {
        hitTest->isHit = true;
        hitTest->materialIndex = record.materialIndex;
    } else {
        hitTest->isHit = false;
    }
}

void hitTest(
    const Scene &scene,
    SceneModel &sceneModel,
    const CUDAGlobals &cudaGlobals,
    int pixelX,
    int pixelY,
    int width,
    int height
) {
    Primitive **dev_primitives;
    Material *dev_materials;
    PrimitiveList **dev_world;

    HitTest *dev_hitTest;

    checkCudaErrors(cudaMalloc((void **)&dev_primitives, primitiveCount * sizeof(Primitive *)));
    checkCudaErrors(cudaMalloc((void **)&dev_materials, materialCount * sizeof(Material)));
    checkCudaErrors(cudaMalloc((void **)&dev_world, sizeof(PrimitiveList *)));
    checkCudaErrors(cudaMalloc((void **)&dev_hitTest, sizeof(HitTest)));

    createWorld<<<1, 1>>>(
        dev_primitives,
        dev_materials,
        dev_world,
        sceneModel.getLightPosition(),
        false
    );

    checkCudaErrors(cudaGetLastError());

    dim3 blocks(width, height);
    hitTestKernel<<<blocks, 1>>>(
        pixelX,
        pixelY,
        dev_world,
        cudaGlobals.d_camera,
        dev_hitTest
    );

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    HitTest hitTest;
    checkCudaErrors(cudaMemcpy(&hitTest, dev_hitTest, sizeof(HitTest), cudaMemcpyDeviceToHost));

    if (hitTest.isHit) {
        const Material &material = scene.getMaterial(hitTest.materialIndex);
        sceneModel.setMaterialIndex(hitTest.materialIndex, material.getAlbedo());
    } else {
        sceneModel.setMaterialIndex(-1, Vec3(0.f));
    }

    checkCudaErrors(cudaFree(dev_hitTest));
}

}

#undef checkCudaErrors
