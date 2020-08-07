#include "materials/material.h"

namespace rays {

__device__ Vec3 Material::getEmit(const HitRecord &hit) const
{
    if (hit.isFront()) {
        return m_emit;
    }
    return Vec3(0.f);
}

__host__ void Material::setEmit(const Vec3 &emit)
{
    m_emit = emit;
}

__host__ void Material::setAlbedo(const Vec3 &albedo)
{
    m_albedo = albedo;
}

}
