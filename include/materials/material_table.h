#pragma once

#include <vector>

#include "materials/material.h"
#include "materials/mirror.h"
#include "materials/types.h"

namespace rays {

struct MaterialLookup {
    MaterialIndex *indices;

    Material *lambertians;
    Mirror *mirrors;
};

class MaterialTable {
public:
    MaterialIndex addMaterial(const Material &material);
    MaterialIndex addMaterial(const Mirror &material);

    void updateMaterial(int materialIndex, Material material) {
        m_lambertians[materialIndex] = material;
    }

    const std::vector<Material> &getLambertians() const {
        return m_lambertians;
    }

    const std::vector<Mirror> &getMirrors() const {
        return m_mirrors;
    }

private:
    std::vector<Material> m_lambertians;
    std::vector<Mirror> m_mirrors;
};

}
