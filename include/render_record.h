#pragma once

#include <cuda_runtime.h>

namespace rays {

struct RenderRecord {
    cudaEvent_t beginEvent;
    cudaEvent_t endEvent;
};

}
