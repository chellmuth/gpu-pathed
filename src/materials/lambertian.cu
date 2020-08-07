#include "materials/lambertian.h"

namespace rays {

__device__ Vec3 Lambertian::getEmit(const HitRecord &hit) const
{
    if (hit.isFront()) {
        return m_emit;
    }
    return Vec3(0.f);
}

__host__ void Lambertian::setEmit(const Vec3 &emit)
{
    m_emit = emit;
}

__host__ void Lambertian::setAlbedo(const Vec3 &albedo)
{
    m_albedo = albedo;
}

}
