#pragma once

#include <vector>

#include "materials/material.h"
#include "materials/material_table.h"
#include "materials/mirror.h"
#include "materials/types.h"

namespace rays {

class MaterialStore {
public:
    int addMaterial(const Material &material);
    int addMaterial(const Mirror &material);

private:
    std::vector<MaterialIndex> m_indices;
    MaterialTable m_table;
};

}
