#pragma once

namespace rays {

enum class MaterialType {
    Lambertian,
    Mirror
};

struct MaterialIndex {
    MaterialType materialType;
    size_t index;
};

}
