#pragma once

#include <iostream>

#include <cuda_runtime.h>
#include <optix.h>

namespace rays {

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess) {
        std::cerr << "CUDA error: " <<  cudaGetErrorString(code)
                  << " file: " << file << " line: " << line
                  << std::endl;

		if (abort) { exit(code); }
	}
}

inline void optixAssert(OptixResult code, const char *file, int line, bool abort = true)
{
	if (code != OPTIX_SUCCESS) {
        std::cerr << "Optix error file: " << file
                  << " code: " << code
                  << " line: " << line
                  << std::endl;

		if (abort) { exit(code); }
	}
}

}
