#pragma once

#include <vector>

#include "materials/glass.h"
#include "materials/lambertian.h"
#include "materials/material_table.h"
#include "materials/mirror.h"
#include "materials/types.h"

namespace rays {

class MaterialStore {
public:
    int addMaterial(const Lambertian &material);
    int addMaterial(const Mirror &material);
    int addMaterial(const Glass &material);

    MaterialIndex indexAt(int materialID) const {
        return m_indices[materialID];
    }

    void updateIndex(int materialID, MaterialIndex newIndex) {
        m_indices[materialID] = newIndex;
    }

    void updateMaterial(int materialIndex, Lambertian material) {
        m_table.updateMaterial(materialIndex, material);
    }

    const std::vector<MaterialIndex> &getIndices() const {
        return m_indices;
    }

    const std::vector<Lambertian> &getLambertians() const {
        return m_table.getLambertians();
    }

    const std::vector<Mirror> &getMirrors() const {
        return m_table.getMirrors();
    }

    const std::vector<Glass> &getGlasses() const {
        return m_table.getGlasses();
    }

private:
    std::vector<MaterialIndex> m_indices;
    MaterialTable m_table;
};

}
