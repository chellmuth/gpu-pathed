#pragma once

#include "vec3.h"

namespace rays {

struct SurfaceSample {
    Vec3 point;
    Vec3 normal;
    float pdf;
};


}
