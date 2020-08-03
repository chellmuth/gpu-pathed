#include "materials/material_table.h"

namespace rays {

MaterialIndex MaterialTable::addMaterial(const Material &material)
{
    m_lambertians.push_back(material);
    return MaterialIndex{
        MaterialType::Lambertian,
        m_lambertians.size() - 1
    };
}

MaterialIndex MaterialTable::addMaterial(const Dummy &material)
{
    m_dummies.push_back(material);
    return MaterialIndex{
        MaterialType::Dummy,
        m_dummies.size() - 1
    };
}

}
