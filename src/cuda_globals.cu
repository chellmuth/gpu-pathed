#include "cuda_globals.h"

#include <iostream>

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

}

#undef checkCudaErrors
