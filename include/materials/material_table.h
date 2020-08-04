#pragma once

#include <vector>

#include "materials/material.h"
#include "materials/mirror.h"
#include "materials/types.h"

namespace rays {

struct MaterialLookup {
    Material *lambertians;
    size_t lambertianSize;

    Mirror *mirrors;
    size_t mirrorSize;
};

struct MaterialTableOffsets {
    size_t getOffset(MaterialType type) {
        switch(type) {
        case MaterialType::Lambertian: {
            return lambertian;
        }
        case MaterialType::Mirror: {
            return mirror;
        }
        }
        return 0;
    }

    size_t lambertian;
    size_t mirror;
};

class MaterialTable {
public:
    MaterialIndex addMaterial(const Material &material);
    MaterialIndex addMaterial(const Mirror &material);

    const std::vector<Material> &getLambertians() const {
        return m_lambertians;
    }

    const std::vector<Mirror> &getMirrors() const {
        return m_mirrors;
    }

    MaterialTableOffsets getOffsets() {
        return MaterialTableOffsets {
            m_lambertians.size(),
            m_mirrors.size()
        };
    }

private:
    std::vector<Material> m_lambertians;
    std::vector<Mirror> m_mirrors;
};

}
