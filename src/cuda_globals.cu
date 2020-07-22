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
    Sphere *spheres,
    Material *materials
) {
    if (blockIdx.x != 0 || blockIdx.y != 0) { return; }
    if (threadIdx.x != 0 || threadIdx.y != 0) { return; }

    *world = PrimitiveList(
        triangles,
        triangleCount,
        spheres,
        sphereCount,
        materials,
        materialCount
    );
}

void CUDAGlobals::mallocWorld()
{
    checkCudaErrors(cudaMalloc((void **)&d_materials, materialCount * sizeof(Material)));

    checkCudaErrors(cudaMalloc((void **)&d_triangles, triangleCount * sizeof(Triangle)));
    checkCudaErrors(cudaMalloc((void **)&d_spheres, sphereCount * sizeof(Sphere)));

    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(PrimitiveList)));

    initWorldKernel<<<1, 1>>>(d_world, d_triangles, d_spheres, d_materials);
    checkCudaErrors(cudaDeviceSynchronize());

}


}

#undef checkCudaErrors
