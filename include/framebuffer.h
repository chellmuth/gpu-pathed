#pragma once

#include <cuda.h>

#include "vec3.h"

namespace rays {

void updateFramebuffer(
    uchar4 *d_fb,
    Vec3 *d_passRadiances,
    Vec3 *d_radiances,
    int passSamples,
    int previousSamples,
    int width, int height,
    CUstream stream
);

}
