#pragma once

#include "core/vec3.h"

namespace rays {

struct SurfaceSample {
    Vec3 point;
    Vec3 normal;
    float pdf;
};


}
