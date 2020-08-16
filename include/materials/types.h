#pragma once

namespace rays {

enum class MaterialType {
    Lambertian = 0,
    Mirror = 1,
    Glass = 2,
    Microfacet = 3,
    None
};

struct MaterialIndex {
    MaterialType materialType;
    size_t index;
};

}
