#pragma once

namespace rays {

enum class MaterialType {
    Lambertian,
    Dummy
};

struct MaterialIndex {
    MaterialType materialType;
    size_t index;
};

}
