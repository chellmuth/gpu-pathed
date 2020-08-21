#pragma once

namespace rays {

enum class PrimitiveType {
    Triangle,
    Sphere
};

struct PrimitiveIndex {
    PrimitiveType primitiveType;
    int index;
};

}
