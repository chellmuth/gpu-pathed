#pragma once

#include "vec3.h"

namespace rays {

struct BSDFSample {
    Vec3 wiLocal;
    float pdf;
    Vec3 f;
    bool isDelta;
};

}
