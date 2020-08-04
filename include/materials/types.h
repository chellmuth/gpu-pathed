#pragma once

namespace rays {

enum class MaterialType {
    Lambertian = 0,
    Mirror = 1
};

struct MaterialIndex {
    MaterialType materialType;
    size_t index;
};

}
