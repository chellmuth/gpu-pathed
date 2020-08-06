#include "materials/material_store.h"

namespace rays {

int MaterialStore::addMaterial(const Material &material)
{
    MaterialIndex index = m_table.addMaterial(material);
    m_indices.push_back(index);
    return m_indices.size() - 1;
}

int MaterialStore::addMaterial(const Mirror &material)
{
    MaterialIndex index = m_table.addMaterial(material);
    m_indices.push_back(index);
    return m_indices.size() - 1;
}

int MaterialStore::addMaterial(const Glass &material)
{
    MaterialIndex index = m_table.addMaterial(material);
    m_indices.push_back(index);
    return m_indices.size() - 1;
}

}
