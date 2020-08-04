#include "hit_test.h"

#include <cfloat>
#include <iostream>
#include <memory>

#include "camera.h"
#include "macro_helper.h"
#include "materials/material.h"
#include "materials/types.h"
#include "primitive.h"
#include "scene.h"
#include "vec3.h"

#define checkCudaErrors(result) { gpuAssert((result), __FILE__, __LINE__); }

namespace rays {


struct HitTest {
    int materialID;
    bool isHit;
};

__global__ static void hitTestKernel(
    int pixelX,
    int pixelY,
    PrimitiveList *world,
    Camera *camera,
    HitTest *hitTest
) {
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int col = threadIdx.x + blockIdx.x * blockDim.x;

    if ((col != pixelX) || (row != pixelY)) { return; }

    const Ray cameraRay = camera->generateRay(row, col);
    HitRecord record;

    bool isHit = world->hit(cameraRay, 0.f, FLT_MAX, record);
    if (isHit) {
        hitTest->isHit = true;
        hitTest->materialID = record.materialID;
    } else {
        hitTest->isHit = false;
    }
}

void hitTest(
    SceneModel &sceneModel,
    const CUDAGlobals &cudaGlobals,
    int pixelX,
    int pixelY,
    int width,
    int height
) {
    HitTest *dev_hitTest;
    checkCudaErrors(cudaMalloc((void **)&dev_hitTest, sizeof(HitTest)));

    dim3 blocks(width, height);
    hitTestKernel<<<blocks, 1>>>(
        pixelX,
        pixelY,
        cudaGlobals.d_world,
        cudaGlobals.d_camera,
        dev_hitTest
    );

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    HitTest hitTest;
    checkCudaErrors(cudaMemcpy(&hitTest, dev_hitTest, sizeof(HitTest), cudaMemcpyDeviceToHost));

    if (hitTest.isHit) {
        sceneModel.setMaterialID(hitTest.materialID);
    } else {
        sceneModel.setMaterialID(-1);
    }

    checkCudaErrors(cudaFree(dev_hitTest));
}

}
