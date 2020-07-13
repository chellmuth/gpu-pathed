#pragma once

#include "primitive.h"
#include "vec3.h"

namespace rays {

constexpr int primitiveCount = 4;

__global__ void createWorld(
    Primitive **primitives,
    PrimitiveList **world,
    Vec3 color,
    float lightPosition,
    bool update
);

}
