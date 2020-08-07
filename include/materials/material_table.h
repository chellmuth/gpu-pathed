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

class MaterialTable {
public:
    MaterialIndex addMaterial(const Lambertian &material);
    MaterialIndex addMaterial(const Mirror &material);
    MaterialIndex addMaterial(const Glass &material);

    void updateMaterial(int materialIndex, Lambertian material) {
        m_lambertians[materialIndex] = material;
    }

    const std::vector<Lambertian> &getLambertians() const {
        return m_lambertians;
    }

    const std::vector<Mirror> &getMirrors() const {
        return m_mirrors;
    }

    const std::vector<Glass> &getGlasses() const {
        return m_glasses;
    }

private:
    std::vector<Lambertian> m_lambertians;
    std::vector<Mirror> m_mirrors;
    std::vector<Glass> m_glasses;
};

}
