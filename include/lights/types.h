#pragma once

#include "core/vec3.h"

namespace rays {

struct LightSample {
    Vec3 wi;
    float distance;
    float pdf;
    Vec3 emitted;
};

enum class PrimitiveType {
    Triangle,
    Sphere
};

struct LightIndex {
    PrimitiveType primitiveType;
    int index;
};

}
