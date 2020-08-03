#pragma once

#include <vector>

#include "materials/material.h"
#include "materials/dummy.h"
#include "materials/types.h"

namespace rays {

struct MaterialLookup {
    Material *lambertians;
    size_t lambertianSize;

    Dummy *dummies;
    size_t dummySize;
};

struct MaterialTableOffsets {
    size_t getOffset(MaterialType type) {
        switch(type) {
        case MaterialType::Lambertian: {
            return lambertian;
        }
        case MaterialType::Dummy: {
            return dummy;
        }
        }
        return 0;
    }

    size_t lambertian;
    size_t dummy;
};

class MaterialTable {
public:
    MaterialIndex addMaterial(const Material &material);
    MaterialIndex addMaterial(const Dummy &material);

    const std::vector<Material> getLambertians() const {
        return m_lambertians;
    }

    MaterialTableOffsets getOffsets() {
        return MaterialTableOffsets {
            m_lambertians.size(),
            m_dummies.size()
        };
    }

private:
    std::vector<Material> m_lambertians;
    std::vector<Dummy> m_dummies;
};

}
