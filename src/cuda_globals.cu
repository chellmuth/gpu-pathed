#include "cuda_globals.h"

#include "macro_helper.h"
#include "scene.h"

#define checkCudaErrors(result) { gpuAssert((result), __FILE__, __LINE__); }

namespace rays {

void CUDAGlobals::mallocCamera()
{
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera)));
}

void CUDAGlobals::copyCamera(const Camera &camera)
{
    checkCudaErrors(cudaMemcpy(
        d_camera,
        &camera,
        sizeof(Camera),
        cudaMemcpyHostToDevice
    ));
}

__global__ static void initWorldKernel(
    PrimitiveList *world,
    Triangle *triangles,
    int triangleSize,
    Sphere *spheres,
    int sphereSize,
    int *lightIndices,
    int lightIndexSize,
    Material *lambertians,
    int lambertianSize,
    MaterialLookup *materialLookup
) {
    if (blockIdx.x != 0 || blockIdx.y != 0) { return; }
    if (threadIdx.x != 0 || threadIdx.y != 0) { return; }

    materialLookup->lambertians = lambertians;
    materialLookup->lambertianSize = lambertianSize;

    *world = PrimitiveList(
        triangles,
        triangleSize,
        spheres,
        sphereSize,
        lightIndices,
        lightIndexSize,
        materialLookup
    );
}

void CUDAGlobals::mallocWorld(const SceneData &sceneData)
{
    const int lambertianSize = sceneData.lambertians.size();
    const int triangleSize = sceneData.triangles.size();
    const int sphereSize = sceneData.spheres.size();
    const int lightIndexSize = sceneData.lightIndices.size();

    checkCudaErrors(cudaMalloc((void **)&d_lambertians, lambertianSize * sizeof(Material)));
    checkCudaErrors(cudaMalloc((void **)&d_dummies, 1 * sizeof(Dummy)));

    checkCudaErrors(cudaMalloc((void **)&d_materialLookup, sizeof(MaterialLookup)));

    checkCudaErrors(cudaMalloc((void **)&d_triangles, triangleSize * sizeof(Triangle)));
    checkCudaErrors(cudaMalloc((void **)&d_spheres, sphereSize * sizeof(Sphere)));
    checkCudaErrors(cudaMalloc((void **)&d_lightIndices, lightIndexSize * sizeof(int)));

    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(PrimitiveList)));

    initWorldKernel<<<1, 1>>>(
        d_world,
        d_triangles,
        triangleSize,
        d_spheres,
        sphereSize,
        d_lightIndices,
        lightIndexSize,
        d_lambertians,
        lambertianSize,
        d_materialLookup
    );
    checkCudaErrors(cudaDeviceSynchronize());
}

void CUDAGlobals::copySceneData(const SceneData &sceneData)
{
    checkCudaErrors(cudaMemcpy(
        d_triangles,
        sceneData.triangles.data(),
        sceneData.triangles.size() * sizeof(Triangle),
        cudaMemcpyHostToDevice
    ));

    checkCudaErrors(cudaMemcpy(
        d_spheres,
        sceneData.spheres.data(),
        sceneData.spheres.size() * sizeof(Sphere),
        cudaMemcpyHostToDevice
    ));

    checkCudaErrors(cudaMemcpy(
        d_lightIndices,
        sceneData.lightIndices.data(),
        sceneData.lightIndices.size() * sizeof(int),
        cudaMemcpyHostToDevice
    ));
}

}
