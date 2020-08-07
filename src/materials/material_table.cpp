#include "materials/material_table.h"

namespace rays {

MaterialIndex MaterialTable::addMaterial(const Lambertian &material)
{
    m_lambertians.push_back(material);
    return MaterialIndex{
        MaterialType::Lambertian,
        m_lambertians.size() - 1
    };
}

MaterialIndex MaterialTable::addMaterial(const Mirror &material)
{
    m_mirrors.push_back(material);
    return MaterialIndex{
        MaterialType::Mirror,
        m_mirrors.size() - 1
    };
}

MaterialIndex MaterialTable::addMaterial(const Glass &material)
{
    m_glasses.push_back(material);
    return MaterialIndex{
        MaterialType::Glass,
        m_glasses.size() - 1
    };
}

}
