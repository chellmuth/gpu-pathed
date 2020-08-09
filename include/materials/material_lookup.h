#pragma once

#include <vector>

#include "materials/glass.h"
#include "materials/lambertian.h"
#include "materials/mirror.h"
#include "materials/types.h"

namespace rays {

struct MaterialLookup {
    MaterialIndex *indices;

    Lambertian *lambertians;
    Mirror *mirrors;
    Glass *glasses;
};

}
