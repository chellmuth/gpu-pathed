#pragma once

#include "core/vec3.h"
#include "primitives/types.h"

namespace rays {

struct LightSample {
    Vec3 wi;
    Vec3 normal;
    float distance;
    float pdf;
    Vec3 emitted;
};

// fixme Remove and use PrimitiveIndex
struct LightIndex {
    PrimitiveType primitiveType;
    int index;
};

}
