#pragma once

#include "material.h"
#include "primitive.h"
#include "vec3.h"

namespace rays {

constexpr int primitiveCount = 4;
constexpr int materialCount = 3;

__global__ void createWorld(
    Primitive **primitives,
    Material *materials,
    PrimitiveList **world,
    Vec3 color,
    float lightPosition,
    bool update
);

}
