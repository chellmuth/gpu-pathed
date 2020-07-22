#include "cuda_globals.h"

#include <iostream>

#include "scene.h"

#define checkCudaErrors(result) { gpuAssert((result), __FILE__, __LINE__); }
static void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess) {
		fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

namespace rays {

void CUDAGlobals::copyCamera(const Camera &camera)
{
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera)));

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
    Material *materials,
    int materialSize
) {
    if (blockIdx.x != 0 || blockIdx.y != 0) { return; }
    if (threadIdx.x != 0 || threadIdx.y != 0) { return; }

    *world = PrimitiveList(
        triangles,
        triangleSize,
        spheres,
        sphereSize,
        materials,
        materialSize
    );
}

void CUDAGlobals::mallocWorld(const SceneData &sceneData)
{
    const int materialSize = sceneData.materials.size();
    const int triangleSize = sceneData.triangles.size();
    const int sphereSize = sceneData.spheres.size();

    checkCudaErrors(cudaMalloc((void **)&d_materials, materialSize * sizeof(Material)));

    checkCudaErrors(cudaMalloc((void **)&d_triangles, triangleSize * sizeof(Triangle)));
    checkCudaErrors(cudaMalloc((void **)&d_spheres, sphereSize * sizeof(Sphere)));

    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(PrimitiveList)));

    initWorldKernel<<<1, 1>>>(
        d_world,
        d_triangles,
        triangleSize,
        d_spheres,
        sphereSize,
        d_materials,
        materialSize
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
}

}

#undef checkCudaErrors
